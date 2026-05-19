"""
Qdrant Vector DB Module - Kết nối, lưu trữ và truy vấn vector trên Qdrant.
Hỗ trợ: local mode (offline) và cloud mode.

Đồng nhất với upsert_data.ipynb:
- HNSW index với m=16, ef_construct=100, full_scan_threshold=10000
- Payload indices: 4 trường (keyword/integer) cho weekly-chunked data
- UUID v5 làm point ID (collision-resistant, deterministic)
- Payload structure (weekly chunks): nested metadata (id/content/original_id/metadata)
  trong đó metadata chứa: week, source_event_ids, timestamp_unix_min, timestamp_unix_max
- Per-event fields (event_name, chairperson, participants, scope...) KHÔNG còn trong payload
  — chúng nằm trong text content, được tìm kiếm qua vector + BM25 hybrid search
"""
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    from ..ingestion.chunker import Chunk
except ImportError:
    from ingestion.chunker import Chunk
from .embedding import embed_texts, embed_query, get_embedding_dimension

try:
    from ..config import (
        QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
        QDRANT_DISTANCE, QDRANT_BATCH_SIZE,
    )
except ImportError:
    from config import (
        QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
        QDRANT_DISTANCE, QDRANT_BATCH_SIZE,
    )

from ..middleware.retry import with_retry

logger = logging.getLogger(__name__)


# Các trường metadata cần tạo index — đồng nhất với upsert_data.ipynb (weekly chunks)
PAYLOAD_INDEX_FIELDS = [
    # Keyword fields: exact match
    ("metadata.week",                  models.KeywordIndexType.KEYWORD,   None),
    ("metadata.source_event_ids",       models.KeywordIndexType.KEYWORD,   None),
    # Integer fields: dùng range query (timestamp)
    ("metadata.timestamp_unix_min",     models.IntegerIndexType.INTEGER,  None),
    ("metadata.timestamp_unix_max",     models.IntegerIndexType.INTEGER,  None),
]

# HNSW config — đồng nhất với upsert_data.ipynb
HNSW_M = 16


# Map từ temporal detection format ("Thứ 2") → JSONL format ("Thứ Hai")
# Dùng case-insensitive matching để xử lý cả "Thứ 2" lẫn "thứ 2"
_DOW_NORMALIZE: dict[str, str] = {
    "thứ 2": "Thứ Hai", "thứ hai": "Thứ Hai",
    "thứ 3": "Thứ Ba", "thứ ba": "Thứ Ba",
    "thứ 4": "Thứ Tư", "thứ tư": "Thứ Tư",
    "thứ 5": "Thứ Năm", "thứ năm": "Thứ Năm",
    "thứ 6": "Thứ Sáu", "thứ sáu": "Thứ Sáu",
    "thứ 7": "Thứ Bảy", "thứ bảy": "Thứ Bảy",
    "chủ nhật": "Chủ Nhật",
    "cn": "Chủ Nhật",
}


def _normalize_day_of_week(dow: str | None) -> str | None:
    """Chuẩn hóa day_of_week: temporal format → JSONL format (exact match cho Qdrant KEYWORD)."""
    if dow is None:
        return None
    return _DOW_NORMALIZE.get(dow.lower(), dow)
HNSW_EF_CONSTRUCT = 100
HNSW_FULL_SCAN_THRESHOLD = 10000


@dataclass
class QdrantConfig:
    """Cấu hình kết nối Qdrant Cloud."""
    url: str = QDRANT_URL
    api_key: str = QDRANT_API_KEY
    collection_name: str = QDRANT_COLLECTION
    vector_size: int | None = None  # auto-detect nếu None
    distance: str = QDRANT_DISTANCE

    def __post_init__(self):
        if self.vector_size is None:
            try:
                self.vector_size = get_embedding_dimension()
                logger.info(f"Auto-detect vector_size: {self.vector_size}")
            except Exception:
                self.vector_size = 1024


class QdrantManager:
    """Quản lý kết nối, index và query trên Qdrant."""

    def __init__(self, config: QdrantConfig | None = None):
        self.config = config or QdrantConfig()
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=30.0,
            )
        return self._client

    def create_collection(self, recreate: bool = False) -> None:
        """
        Tạo collection với schema và HNSW index.
        Cải tiến từ upsert_data.ipynb: cấu hình HNSW đầy đủ ngay khi tạo.
        """
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclidean": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        distance = distance_map.get(self.config.distance, models.Distance.COSINE)

        if recreate and self.collection_exists():
            logger.info(f"Xoá collection cũ: {self.config.collection_name}")
            self.client.delete_collection(collection_name=self.config.collection_name)

        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.vector_size or 1024,
                    distance=distance,
                    hnsw_config=models.HnswConfigDiff(
                        m=HNSW_M,
                        ef_construct=HNSW_EF_CONSTRUCT,
                        full_scan_threshold=HNSW_FULL_SCAN_THRESHOLD,
                    ),
                ),
            )
            logger.info(
                f"Đã tạo collection: {self.config.collection_name} "
                f"(HNSW: m={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCT}, "
                f"full_scan_threshold={HNSW_FULL_SCAN_THRESHOLD})"
            )
        else:
            logger.info(f"Collection đã tồn tại: {self.config.collection_name}")

    def collection_exists(self) -> bool:
        """Kiểm tra collection có tồn tại không."""
        try:
            self.client.get_collection(collection_name=self.config.collection_name)
            return True
        except (UnexpectedResponse, Exception):
            return False

    def setup_hnsw_index(self) -> None:
        """
        Cấu hình HNSW index sau khi collection đã tồn tại.
        Dùng khi collection đã tạo mà chưa có HNSW config.
        """
        if not self.collection_exists():
            return
        try:
            self.client.update_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    "": models.VectorParamsDiff(
                        hnsw_config=models.HnswConfigDiff(
                            m=HNSW_M,
                            ef_construct=HNSW_EF_CONSTRUCT,
                            full_scan_threshold=HNSW_FULL_SCAN_THRESHOLD,
                        )
                    )
                },
            )
            logger.info("HNSW index configured")
        except Exception as e:
            logger.warning(f"Không cấu hình được HNSW: {e}")

    def ensure_indexes(self) -> None:
        """
        Tạo payload indexes cho các trường metadata.
        Đồng nhất với upsert_data.ipynb (weekly-chunked payload):
        - 2 keyword fields: week, source_event_ids
        - 2 integer fields: timestamp_unix_min, timestamp_unix_max

        Không làm gì nếu index đã tồn tại. Không cần recreate collection.
        """
        if not self.collection_exists():
            return

        try:
            info = self.client.get_collection(self.config.collection_name)
            existing = getattr(info, "payload_schema", {}) or {}

            for field_name, index_type, index_params in PAYLOAD_INDEX_FIELDS:
                if field_name not in existing:
                    if index_type == models.KeywordIndexType.KEYWORD:
                        self.client.create_payload_index(
                            collection_name=self.config.collection_name,
                            field_name=field_name,
                            field_schema=models.KeywordIndexParams(
                                type=models.KeywordIndexType.KEYWORD,
                            ),
                        )
                    elif index_type == models.TextIndexType.TEXT:
                        # Text fields: dùng TextIndexParams đã định nghĩa sẵn
                        self.client.create_payload_index(
                            collection_name=self.config.collection_name,
                            field_name=field_name,
                            field_schema=index_params,
                        )
                    elif index_type == models.IntegerIndexType.INTEGER:
                        self.client.create_payload_index(
                            collection_name=self.config.collection_name,
                            field_name=field_name,
                            field_schema=models.IntegerIndexParams(
                                type=models.IntegerIndexType.INTEGER,
                            ),
                        )
                    else:
                        # Fallback: dùng index_params nếu có
                        if index_params is not None:
                            self.client.create_payload_index(
                                collection_name=self.config.collection_name,
                                field_name=field_name,
                                field_schema=index_params,
                            )
                    logger.info(f"Created index: {field_name} ({index_type.name})")
                else:
                    logger.debug(f"Index already exists: {field_name}")
        except Exception as e:
            logger.warning(f"Không tạo được indexes: {e}")

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = QDRANT_BATCH_SIZE) -> int:
        """
        Đẩy chunks lên Qdrant theo batch.
        Đồng nhất với upsert_data.ipynb (weekly-chunked payload):
        - Payload structure: nested metadata (id/content/original_id/metadata)
        - Dùng chunk.id (UUID v5) làm point ID
        - timestamp_unix_min/max (ms) cho weekly range query
        """
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        points = []
        for i, chunk in enumerate(chunks):
            # Loại bỏ các trường nội bộ khỏi payload metadata
            original_meta = chunk.metadata.copy()
            original_meta.pop("chunk_id", None)
            original_meta.pop("original_id", None)
            # timestamp_unix_min/max (ms) — set if timestamp available
            if "timestamp" in original_meta and original_meta["timestamp"] is not None:
                original_meta["timestamp_unix_min"] = int(original_meta["timestamp"]) * 1000
                original_meta["timestamp_unix_max"] = int(original_meta["timestamp"]) * 1000

            # Payload structure đồng nhất với upsert_data.ipynb
            payload = {
                "id": f"{chunk.source_id}_chunk_{chunk.chunk_index}",  # chunk ID gốc
                "content": chunk.text,
                "original_id": chunk.source_id,
                "metadata": original_meta,  # nested metadata
            }

            points.append(models.PointStruct(
                id=chunk.id,   # UUID v5 — deterministic, collision-resistant
                vector=embeddings[i],
                payload=payload,
            ))

        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
            total += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

        logger.info(f"Đã upsert {total} chunks lên Qdrant")
        return total

    def search(
        self,
        query: str,
        top_k: int = 5,
        scope_filter: str | None = None,
        week_filter: str | None = None,
        month_filter: int | None = None,
        year_filter: int | None = None,
        day_filter: int | None = None,
        day_of_week_filter: str | None = None,
        location_filter: str | None = None,
        event_name_filter: str | None = None,
        chairperson_filter: str | None = None,
        participants_filter: str | None = None,
        domain_filter: str | None = None,
        timestamp_from: int | None = None,
        timestamp_to: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm vector, trả về top_k kết quả.
        Đồng nhất với upsert_data.ipynb (weekly-chunked payload).

        Lưu ý: weekly-chunked payload chỉ có 4 indexed fields:
        metadata.week, metadata.source_event_ids, metadata.timestamp_unix_min, metadata.timestamp_unix_max.
        Per-event fields (event_name, chairperson, participants, location, scope, day_of_week...)
        KHÔNG còn trong payload — chúng nằm trong text content, được tìm kiếm qua
        vector + BM25 hybrid search (xử lý ở tầng hybrid_search trong main.py).
        """
        query_vector = embed_query(query)

        filter_conditions = []

        # --- Weekly-chunked payload filters (CÓ trong Qdrant) ---
        if week_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.week",
                    match=models.MatchValue(value=week_filter),
                )
            )

        # timestamp filter: dùng timestamp_unix_min/max (weekly chunks)
        if timestamp_from is not None or timestamp_to is not None:
            range_cond = {}
            if timestamp_from is not None:
                range_cond["gte"] = timestamp_from
            if timestamp_to is not None:
                range_cond["lte"] = timestamp_to
            # Lọc chunk mà event range overlap với query range
            # Chunk match nếu: chunk.min <= query.to AND chunk.max >= query.from
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.timestamp_unix_max",
                    range=models.Range(gte=timestamp_from) if timestamp_from else None,
                )
            )
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.timestamp_unix_min",
                    range=models.Range(lte=timestamp_to) if timestamp_to else None,
                )
            )

        # --- Per-event field filters (KHÔNG có trong weekly-chunked payload, bỏ qua) ---
        # Các filters dưới đây được giữ lại trong signature để caller không lỗi,
        # nhưng chúng không áp dụng được vì fields không tồn tại trong payload.
        # Xử lý ở tầng hybrid_search (main.py) bằng BM25 text matching + result post-filter.
        if scope_filter:
            logger.debug(f"scope_filter='{scope_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if month_filter is not None:
            logger.debug(f"month_filter={month_filter} bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if year_filter is not None:
            logger.debug(f"year_filter={year_filter} bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if day_filter is not None:
            logger.debug(f"day_filter={day_filter} bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if day_of_week_filter is not None:
            logger.debug(f"day_of_week_filter='{day_of_week_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if location_filter is not None:
            logger.debug(f"location_filter='{location_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if event_name_filter is not None:
            logger.debug(f"event_name_filter='{event_name_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if chairperson_filter is not None:
            logger.debug(f"chairperson_filter='{chairperson_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if participants_filter is not None:
            logger.debug(f"participants_filter='{participants_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")
        if domain_filter is not None:
            logger.debug(f"domain_filter='{domain_filter}' bị bỏ qua — per-event field, không có trong weekly-chunked payload")

        search_filter = models.Filter(
            must=filter_conditions
        ) if filter_conditions else None

        # A3: retry wrapper cho Qdrant API call
        def _do_query():
            return self.client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
            )

        results = with_retry(_do_query, max_attempts=3, initial_delay=0.5, backoff_factor=2.0)()

        parsed = []
        for r in results.points:
            payload = r.payload or {}
            # Đọc từ nested metadata (weekly-chunked payload structure)
            meta = payload.get("metadata", {})
            parsed.append({
                "id": r.id,
                "score": r.score,
                "text": payload.get("content") or payload.get("text", ""),
                "source_id": payload.get("original_id") or payload.get("source_id", ""),
                # Weekly-chunked fields (CÓ trong payload)
                "week": meta.get("week", ""),
                "timestamp_unix_min": meta.get("timestamp_unix_min"),
                "timestamp_unix_max": meta.get("timestamp_unix_max"),
                "source_event_ids": meta.get("source_event_ids"),
                # Per-event fields (KHÔNG có trong payload mới, để None)
                "scope": None,
                "month": None,
                "year": None,
                "day": None,
                "day_of_week": None,
                "location": None,
                "timestamp_unix": None,
                "url": None,
                "event_name": None,
                "domain": None,
                "participants": None,
                "chairperson": None,
                "metadata": meta,
            })
        return parsed

    def upsert_payloads(
        self,
        updates: list[dict[str, Any]],
    ) -> int:
        """
        Cập nhật payload cho các điểm đã tồn tại mà KHÔNG re-embed.
        updates: list of {"id": int, "payload": dict} — id là point ID.
        Dùng set_payload nên các trường mới được thêm, cũ giữ nguyên.
        """
        total = 0
        batch_size = 200
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            point_ids = [u["id"] for u in batch]
            merged: dict[int, dict] = {}
            for u in batch:
                pid = u["id"]
                if pid not in merged:
                    merged[pid] = {}
                merged[pid].update(u["payload"])
            self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=merged,
                points=point_ids,
            )
            total += len(batch)

        logger.info(f"Đã set {total} payload updates (không re-embed)")
        return total

    def delete_collection(self) -> None:
        """Xoá collection."""
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.config.collection_name)
            logger.info(f"Đã xoá collection: {self.config.collection_name}")

    def get_collection_info(self) -> dict[str, Any]:
        """Lấy thông tin collection."""
        if not self.collection_exists():
            return {"exists": False}
        info = self.client.get_collection(collection_name=self.config.collection_name)
        return {
            "exists": True,
            "name": self.config.collection_name,
            "vectors_count": getattr(info, 'vectors_count', info.points_count),
            "points_count": info.points_count,
            "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
        }

    def get_available_weeks(self) -> list[str]:
        """Lấy danh sách các tuần có trong collection (dùng để resolve 'current'/'next')."""
        if not self.collection_exists():
            return []
        weeks: set[str] = set()
        offset: str | None = None
        while True:
            result = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["metadata.week"],
            )
            for point in result[0]:
                w = point.payload.get("metadata", {}).get("week")
                if w:
                    weeks.add(str(w))
            offset = result[1]
            if offset is None:
                break
        sorted_weeks = sorted(weeks, key=lambda w: self._week_sort_key(w))
        logger.debug(f"Available weeks: {sorted_weeks}")
        return sorted_weeks

    def get_latest_week_metadata(self) -> dict[str, Any] | None:
        """
        Lấy month/year của tuần mới nhất trong collection.
        Trả về {'month': int, 'year': int} hoặc None nếu không có dữ liệu.
        """
        if not self.collection_exists():
            return None
        import re
        weeks = self.get_available_weeks()
        if not weeks:
            return None
        latest_week = weeks[-1]
        wm = re.search(r"W?(\d+)", latest_week, re.IGNORECASE)
        if not wm:
            return None

        offset: str | None = None
        month_val, year_val = None, None
        while True:
            result = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["metadata.week", "metadata.month", "metadata.year"],
            )
            for point in result[0]:
                meta = point.payload.get("metadata", {})
                if meta.get("week") == latest_week:
                    if month_val is None:
                        month_val = meta.get("month")
                        year_val = meta.get("year")
                    if month_val is not None and year_val is not None:
                        offset = None
                        break
            offset = result[1]
            if offset is None:
                break
            if month_val is not None and year_val is not None:
                break

        if month_val is not None and year_val is not None:
            logger.debug(f"Latest week metadata: week={latest_week}, month={month_val}, year={year_val}")
            return {"month": int(month_val), "year": int(year_val)}

        wn = int(wm.group(1))
        year_inferred = 2025 if wn <= 17 else 2026
        logger.debug(f"Latest week metadata (inferred): week={latest_week}, year={year_inferred}")
        return {"month": None, "year": year_inferred}

    @staticmethod
    def _week_sort_key(week_str: str) -> tuple:
        """Sort key cho week string: extract số, sort theo số."""
        import re
        m = re.search(r"W?(\d+)", week_str, re.IGNORECASE)
        if m:
            return (0, int(m.group(1)))
        return (1, week_str)

    def save_local(self, storage_path: str) -> str:
        """Lưu Qdrant local snapshot ra disk."""
        snapshot_info = self.client.create_snapshot(
            collection_name=self.config.collection_name,
        )
        path = Path(storage_path)
        path.mkdir(parents=True, exist_ok=True)
        dest = path / f"{self.config.collection_name}.snapshot"
        logger.info(f"Snapshot saved: {snapshot_info.name} → {dest}")
        return str(dest)
