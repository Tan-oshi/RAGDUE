"""
Main RAG Pipeline - Luồng RAG tổng thể kết nối toàn bộ hệ thống.
Kết hợp: Ingestion → Retrieval (Vector + BM25 hybrid) → Generation (Gemini 2.5).
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer


from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .ingestion import load_records_direct, Chunk
from .ingestion.chunker import ChunkConfig
from .ingestion.weekly_chunker import build_weekly_chunks, WeeklyChunk, get_per_event_metadata
from .retrieval import QdrantConfig, QdrantManager, BM25Retriever
from .retrieval.qdrant_db import _normalize_day_of_week
from .retrieval.temporal import (
    build_recency_boost,
    resolve_week_filter,
    resolve_day_filter,
    resolve_month_filter,
    resolve_year_filter,
    resolve_timestamp_filter,
    TemporalSearchConfig,
    TemporalIntent,
)
from .generation import (
    generate_answer,
    generate_answer_with_history,
    format_context_from_results,
    expand_query_from_history,
)
from .query_parser import QueryParser, ParsedQuery
from .middleware.guardrails import get_guardrails, GuardrailsDecision

try:
    from .config import (
        DEFAULT_ALPHA, DEFAULT_TOP_K, SEMANTIC_MIN_CHUNK, SEMANTIC_MAX_CHUNK,
        QDRANT_BATCH_SIZE, EMBEDDING_MODEL, GEMINI_MODEL,
        TEMPORAL_ENABLE_RECENCY_BOOST, TEMPORAL_MAX_BOOST_FACTOR, TEMPORAL_RECENCY_BOOST_WEIGHT,
        QUERY_CACHE_TTL_SECONDS, QUERY_CACHE_MAX_SIZE,
        RRF_K,
    )
except ImportError:
    from config import (
        DEFAULT_ALPHA, DEFAULT_TOP_K, SEMANTIC_MIN_CHUNK, SEMANTIC_MAX_CHUNK,
        QDRANT_BATCH_SIZE, EMBEDDING_MODEL, GEMINI_MODEL,
        TEMPORAL_ENABLE_RECENCY_BOOST, TEMPORAL_MAX_BOOST_FACTOR, TEMPORAL_RECENCY_BOOST_WEIGHT,
        QUERY_CACHE_TTL_SECONDS, QUERY_CACHE_MAX_SIZE,
        RRF_K,
    )

load_dotenv()

console = Console()


# ---------------------------------------------------------------------------
# Simple Scope/Location Detection (inline, replaces deprecated temporal.py imports)
# ---------------------------------------------------------------------------

def _detect_scope_intent(query: str) -> str:
    """Detect schedule scope: 'nội bộ' vs 'chung'. Default: 'Lịch làm việc chung'."""
    q = query.lower()
    if re.search(r"lịch\s*nội\s*bộ|lich\s*noi\s*bo", q):
        return "Lịch làm việc nội bộ"
    if re.search(r"lịch\s*chung|lich\s*chung", q):
        return "Lịch làm việc chung"
    return "Lịch làm việc chung"


def _detect_location_intent(query: str) -> str | None:
    """Detect room/location from query. Returns location string or None."""
    q = query.lower()
    # Phòng E101, E202...
    m = re.search(r"phòng\s*E\d+", q)
    if m:
        loc = m.group(0)[6:].strip()
        return loc if loc else None
    # "tại phòng X", "ở phòng X"
    m = re.search(r"(?:tại|ở)\s+phòng\s+([^\s,]+)", q)
    if m:
        loc = m.group(1).rstrip(",")
        if loc and len(loc) >= 2:
            return loc
    # "phòng X" standalone
    m = re.search(r"phòng\s+([^\s,]+?)(?:\s|$|,)", q)
    if m:
        loc = m.group(1).rstrip(",")
        if loc and len(loc) >= 2 and loc not in {"thứ", "th", "bảy", "nhật", "hai", "ba", "tư", "năm", "sáu"}:
            return loc
    # E101, E202 standalone
    m = re.search(r"\bE\d{3}\b", q)
    if m:
        return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Query Cache
# ---------------------------------------------------------------------------

class RetrievalCache:
    """Thread-safe LRU cache cho retrieval results."""

    def __init__(self, ttl_seconds: int = QUERY_CACHE_TTL_SECONDS, max_size: int = QUERY_CACHE_MAX_SIZE):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()

    def _make_key(
        self,
        query: str,
        top_k: int,
        alpha: float,
        week_filter: str | None,
        month_filter: int | None,
        year_filter: int | None,
        day_filter: int | None,
        day_of_week_filter: str | None,
    ) -> str:
        raw = json.dumps({
            "q": query,
            "k": top_k,
            "a": alpha,
            "w": week_filter,
            "m": month_filter,
            "y": year_filter,
            "d": day_filter,
            "dw": day_of_week_filter,
        }, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(
        self,
        query: str,
        top_k: int,
        alpha: float,
        week_filter: str | None,
        month_filter: int | None,
        year_filter: int | None,
        day_filter: int | None,
        day_of_week_filter: str | None,
    ) -> list[dict] | None:
        key = self._make_key(query, top_k, alpha, week_filter, month_filter, year_filter, day_filter, day_of_week_filter)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            results, timestamp = entry
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            logger.info(f"[Cache HIT] query='{query[:60]}...'")
            return results

    def put(
        self,
        query: str,
        top_k: int,
        alpha: float,
        week_filter: str | None,
        month_filter: int | None,
        year_filter: int | None,
        day_filter: int | None,
        day_of_week_filter: str | None,
        results: list[dict],
    ):
        key = self._make_key(query, top_k, alpha, week_filter, month_filter, year_filter, day_filter, day_of_week_filter)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (results, time.time())
            logger.info(f"[Cache PUT] query='{query[:60]}...', size={len(self._cache)}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag_due")


@dataclass
class RAGPipeline:
    """Toàn bộ pipeline RAG."""

    jsonl_path: str = ""
    qdrant_config: QdrantConfig = field(default_factory=QdrantConfig)
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    embedding_model: str = EMBEDDING_MODEL
    gemini_model: str = GEMINI_MODEL

    _qdrant: QdrantManager | None = field(default=None, repr=False)
    _bm25: BM25Retriever = field(default_factory=BM25Retriever, repr=False)
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _weekly_chunks: list[WeeklyChunk] = field(default_factory=list, repr=False)
    _per_event_index: dict[str, dict] = field(default_factory=dict, repr=False)
    _loaded: bool = False
    _available_weeks: list[str] = field(default_factory=list, repr=False)
    _cache: RetrievalCache = field(default_factory=RetrievalCache, repr=False)
    _query_parser: QueryParser = field(default=None, repr=False)
    # B4: Conversational memory — rolling window per session
    _session_history: list[dict[str, str]] = field(default_factory=list, repr=False)
    _history_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _max_history: int = 5

    def __post_init__(self):
        self._qdrant = QdrantManager(config=self.qdrant_config)
        self._bm25 = BM25Retriever()
        self._cache = RetrievalCache()
        self._query_parser = QueryParser()

    def _ensure_loaded(self):
        """
        Load JSONL → weekly chunks → build BM25 index, cache available weeks.
        Dùng weekly chunks (đồng nhất với Qdrant collection) để BM25 ID match được với Qdrant IDs.
        Per-event index dùng để post-filter bằng per-event metadata fields.
        """
        if self._loaded:
            return
        import pathlib
        _project_root = pathlib.Path(__file__).resolve().parent.parent
        # Dùng file đã được enrich đầy đủ metadata fields (timestamp, event_name, domain, etc.)
        # — đồng nhất với upsert_data.ipynb
        jsonl_default = str(_project_root / "data/processed/master_lich_tuan_with_event_timestamp.jsonl")
        path = self.jsonl_path or jsonl_default
        if not path:
            raise RuntimeError("Cần cung cấp đường dẫn file JSONL")

        # Build weekly chunks (đồng nhất với Qdrant weekly-chunked data)
        weekly_chunks = build_weekly_chunks(path)
        self._weekly_chunks = weekly_chunks
        self._chunks = weekly_chunks  # alias for compatibility

        # Build BM25 index on weekly chunks (IDs match Qdrant weekly chunk IDs)
        self._bm25.build_index(weekly_chunks)

        # Build per-event index for post-filtering (scope, month, year, etc.)
        per_events: dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                per_events[record.get("id", "")] = record
        self._per_event_index = per_events

        self._loaded = True
        # Pre-load embedding model to avoid cold start on first query
        from .retrieval.embedding import get_embedding_model
        get_embedding_model()
        # Đảm bảo payload indexes tồn tại (không ảnh hưởng vector index)
        try:
            self._qdrant.ensure_indexes()
        except Exception as e:
            logger.warning(f"Không tạo được indexes: {e}")
        try:
            self._available_weeks = self._qdrant.get_available_weeks()
        except Exception:
            self._available_weeks = []
        logger.info(f"Pipeline loaded: {len(weekly_chunks)} weekly chunks, BM25 ready, {len(per_events)} per-event records indexed")

    def ingest(
        self,
        jsonl_path: str | None = None,
        recreate_collection: bool = False,
    ) -> int:
        """
        Ingestion pipeline: JSONL → Chunking → Vector DB (Qdrant) + BM25.
        """
        path = jsonl_path
        if not path:
            import pathlib
            _project_root = pathlib.Path(__file__).resolve().parent.parent
            path = str(_project_root / "data/processed/master_lich_tuan_with_event_timestamp.jsonl")
            logger.info(f"Using default JSONL path: {path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load JSONL + Chunking — mỗi record = 1 chunk (không semantic split)
            # Giữ nguyên event_name từ JSONL, không enrich thêm
            t1 = progress.add_task("Đang đọc file JSONL...", total=None)
            chunks = load_records_direct(path)
            progress.update(t1, description=f"✓ Đã đọc {len(chunks)} bản ghi")
            self._chunks = chunks

            # Build BM25 index
            t3 = progress.add_task("Đang build BM25 index...", total=None)
            self._bm25.build_index(chunks)
            progress.update(t3, description=f"✓ BM25 index: {self._bm25.corpus_size} docs")

            # Qdrant collection
            t4 = progress.add_task("Đang tạo collection Qdrant...", total=None)
            self._qdrant.create_collection(recreate=recreate_collection)
            progress.update(t4, description="✓ Collection sẵn sàng")

            # Upsert to Qdrant
            t5 = progress.add_task("Đang đẩy vectors lên Qdrant...", total=len(chunks))
            self._qdrant.upsert_chunks(chunks, batch_size=QDRANT_BATCH_SIZE)
            progress.update(t5, description=f"✓ Đã upsert {len(chunks)} vectors")

        self._loaded = True

        # Cache available weeks from Qdrant for temporal search
        try:
            self._available_weeks = self._qdrant.get_available_weeks()
            logger.info(f"Cached {len(self._available_weeks)} available weeks: {self._available_weeks}")
        except Exception as e:
            logger.warning(f"Không lấy được available weeks: {e}")
            self._available_weeks = []

        console.print(Panel.fit(
            f"[green]✓ Ingestion hoàn tất![/green]\n"
            f"  Chunks: {len(chunks)} | BM25: {self._bm25.corpus_size}",
            title="Kết quả",
        ))
        return len(chunks)

    def hybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        scope_filter: str | None = None,
        week_filter: str | None = None,
        month_filter: int | None = None,
        year_filter: int | None = None,
        day_filter: int | None = None,
        day_of_week_filter: str | None = None,
        original_temporal_intent: TemporalIntent | None = None,
        parsed_query: ParsedQuery | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Hybrid search: kết hợp vector search (Qdrant) + BM25 với RRF fusion.
        Replaces: alpha * v_norm + (1-alpha) * b_norm
        With: RRF(k=60) — no normalization needed, more robust.

        parsed_query: nếu được cung cấp (từ QueryParser), dùng trực tiếp
        thay vì gọi regex detection. Cho phép graceful fallback nếu
        QueryParser chưa được init.
        """
        if not self._loaded:
            raise RuntimeError("Pipeline chưa ingest dữ liệu. Gọi .ingest() trước.")

        # --- Parse query (LLM-based if available, regex fallback otherwise) ---
        if parsed_query is None:
            parsed_query = self._query_parser.parse(query)

        # --- Scope detection ---
        effective_scope_filter = scope_filter
        if not effective_scope_filter:
            effective_scope_filter = _detect_scope_intent(query)
            logger.info(f"Scope detected: {effective_scope_filter}")

        # --- Location detection ---
        effective_location_filter: str | None = _detect_location_intent(query)
        if effective_location_filter:
            logger.info(f"Location detected: {effective_location_filter}")

        # --- Content filter from QueryParser ---
        effective_event_name_filter = parsed_query.content.event_name
        effective_chairperson_filter = parsed_query.content.chairperson
        effective_participants_filter = parsed_query.content.participants
        if any([effective_event_name_filter, effective_chairperson_filter, effective_participants_filter]):
            logger.info(f"Content filter (QueryParser, conf={parsed_query.confidence:.2f}): event={effective_event_name_filter}, chair={effective_chairperson_filter}, part={effective_participants_filter}")

        # --- Temporal intent from QueryParser ---
        intent = original_temporal_intent
        if intent is None and parsed_query.temporal.has_reference:
            intent = TemporalIntent(
                has_temporal_reference=parsed_query.temporal.has_reference,
                temporal_type=parsed_query.temporal.type,
                explicit_year=parsed_query.temporal.year,
                explicit_month=parsed_query.temporal.month,
                explicit_day=parsed_query.temporal.day,
                explicit_day_of_week=parsed_query.temporal.day_of_week,
                explicit_week=parsed_query.temporal.week,
            )
        effective_week_filter = week_filter

        if not effective_week_filter and intent and intent.has_temporal_reference:
            effective_week_filter = resolve_week_filter(intent, self._available_weeks)
            logger.info(f"Temporal detected: {intent.temporal_type} → week_filter={effective_week_filter}")
        elif not effective_week_filter:
            # Graceful fallback: dùng recency boost thay vì raise exception.
            # QueryParser với LLM phát hiện temporal tốt hơn regex,
            # nhưng vẫn có thể miss edge cases → fallback an toàn.
            logger.info("Không phát hiện tham chiếu thời gian cụ thể → dùng recency boost")

        if effective_week_filter is None and intent and intent.temporal_type == "day_of_week":
            from .retrieval.temporal import _get_latest_week_from_db
            latest = _get_latest_week_from_db(self._available_weeks)
            if latest:
                effective_week_filter = latest
                logger.info(f"No explicit week for day_of_week → default to latest: {effective_week_filter}")

        # --- Latest week metadata ---
        latest_week_meta = None
        try:
            latest_week_meta = self._qdrant.get_latest_week_metadata()
        except Exception as e:
            logger.warning(f"Không lấy được latest week metadata: {e}")

        # --- Resolve temporal filters ---
        effective_month_filter = month_filter
        effective_year_filter = year_filter
        effective_day_filter = day_filter
        effective_day_of_week_filter: str | None = day_of_week_filter
        effective_ts_from: int | None = None
        effective_ts_to: int | None = None

        if intent and intent.has_temporal_reference:
            if effective_day_filter is None and effective_day_of_week_filter is None:
                resolved_day, resolved_dow = resolve_day_filter(intent)
                if resolved_day is not None:
                    effective_day_filter = resolved_day
                    logger.info(f"Day filter resolved: {resolved_day}")
                if resolved_dow is not None:
                    effective_day_of_week_filter = resolved_dow
                    logger.info(f"Day-of-week filter resolved: {resolved_dow}")
            if effective_month_filter is None:
                effective_month_filter = resolve_month_filter(
                    intent,
                    latest_week_month=latest_week_meta.get("month") if latest_week_meta else None,
                )
                if intent.temporal_type == "current_month" and effective_month_filter is not None:
                    effective_year_filter = latest_week_meta.get("year") if latest_week_meta else None
            if effective_year_filter is None:
                effective_year_filter = resolve_year_filter(
                    intent,
                    latest_week_year=latest_week_meta.get("year") if latest_week_meta else None,
                )
            if effective_month_filter or effective_year_filter:
                logger.info(f"Month/year filter: month={effective_month_filter}, year={effective_year_filter}")

            ts_from, ts_to = resolve_timestamp_filter(
                intent,
                latest_week_month=latest_week_meta.get("month") if latest_week_meta else None,
                latest_week_year=latest_week_meta.get("year") if latest_week_meta else None,
            )
            if ts_from is not None or ts_to is not None:
                effective_ts_from = ts_from
                effective_ts_to = ts_to
                logger.info(f"Timestamp range: {ts_from} - {ts_to}")

        # --- Cache lookup ---
        cached = self._cache.get(
            query=query,
            top_k=top_k,
            alpha=alpha,
            week_filter=effective_week_filter,
            month_filter=effective_month_filter,
            year_filter=effective_year_filter,
            day_filter=effective_day_filter,
            day_of_week_filter=effective_day_of_week_filter,
        )
        if cached is not None:
            # Apply recency boost on cached results, then return
            if not effective_week_filter and TEMPORAL_ENABLE_RECENCY_BOOST:
                cached = build_recency_boost(cached, TemporalSearchConfig(
                    recency_boost_weight=TEMPORAL_RECENCY_BOOST_WEIGHT,
                    enable_recency_boost=True,
                    max_boost_factor=TEMPORAL_MAX_BOOST_FACTOR,
                ))
            logger.info(f"[Cache HIT] Returning {len(cached)} cached results")
            return cached[:top_k], effective_scope_filter, intent.temporal_type if intent else None

        # --- Parallel retrieval: Qdrant vector + BM25 (concurrent) ---
        # top_k reduced from 80→20 since data is now weekly-chunked (719 total chunks).
        # Internal multiplier 1.5x: retrieve 30 candidates → combine → output 20.
        effective_top_k = top_k if not effective_week_filter else min(top_k * 2, 30)

        v_results_holder: dict = {}
        bm_results_holder: dict = {}

        def _qdrant_search():
            v_results_holder["data"] = self._qdrant.search(
                query=query,
                top_k=effective_top_k * 2,
                scope_filter=effective_scope_filter,
                week_filter=effective_week_filter,
                month_filter=effective_month_filter,
                year_filter=effective_year_filter,
                day_filter=effective_day_filter,
                day_of_week_filter=effective_day_of_week_filter,
                location_filter=effective_location_filter,
                event_name_filter=effective_event_name_filter,
                chairperson_filter=effective_chairperson_filter,
                participants_filter=effective_participants_filter,
                domain_filter="schedule",
                timestamp_from=effective_ts_from,
                timestamp_to=effective_ts_to,
            )

        def _bm25_search():
            bm_results_holder["data"] = self._bm25.search(query=query, top_k=effective_top_k * 2)

        t_qdrant = threading.Thread(target=_qdrant_search)
        t_bm25 = threading.Thread(target=_bm25_search)
        t_qdrant.start()
        t_bm25.start()
        t_qdrant.join()
        t_bm25.join()

        vector_results = v_results_holder.get("data", [])
        bm25_raw = bm_results_holder.get("data", [])

        # --- Per-event filter: kiểm tra source_event_ids qua per-event metadata ---
        def _chunk_passes_per_event_filter(source_event_ids: list[str]) -> bool:
            """Weekly chunk pass filter nếu ÍT NHẤT 1 per-event trong nó match filter."""
            if not source_event_ids:
                return True  # không có event IDs → không filter

            for eid in source_event_ids:
                event = self._per_event_index.get(eid)
                if event is None:
                    continue
                meta = event.get("metadata") or {}

                # scope filter: weekly chunks chứa cả chung + nội bộ → không filter theo scope
                # scope được xử lý ở tầng content filter (3-tier cascade)

                # content filters: event_name, chairperson, participants (TEXT matching)
                if effective_event_name_filter:
                    en = (meta.get("event_name") or event.get("content") or "")
                    txt = (event.get("content") or "")
                    if effective_event_name_filter.lower() not in en.lower() and effective_event_name_filter.lower() not in txt.lower():
                        continue
                if effective_chairperson_filter:
                    cp = (meta.get("chairperson") or "")
                    txt = (event.get("content") or "")
                    if effective_chairperson_filter.lower() not in cp.lower() and effective_chairperson_filter.lower() not in txt.lower():
                        continue
                if effective_participants_filter:
                    pt = (meta.get("participants") or "")
                    txt = (event.get("content") or "")
                    if effective_participants_filter.lower() not in pt.lower() and effective_participants_filter.lower() not in txt.lower():
                        continue

                # PASS: at least one event matches
                return True

            # FAIL: no events matched
            return False

        # --- Build result index keyed by (week, source_event_ids overlap key) ---
        # Mỗi chunk = unique content (weekly chunk). Key = chunk ID.
        # Match: cùng weekly chunk → Qdrant ID ≈ BM25 ID (do weekly chunking giống nhau).
        # Score: kết hợp normalized scores.

        all_chunk_ids: set[str] = set()
        for r in vector_results:
            all_chunk_ids.add(r["id"])
        for bm in bm25_raw:
            all_chunk_ids.add(bm["id"])

        # Normalize scores
        v_scores = {r["id"]: r["score"] for r in vector_results}
        b_scores = {bm["id"]: bm["score"] for bm in bm25_raw}

        # RRF: compute ranks (1-based)
        v_ranks: dict[str, int] = {}
        for rank, r in enumerate(sorted(vector_results, key=lambda x: x["score"], reverse=True), 1):
            v_ranks[r["id"]] = rank
        b_ranks: dict[str, int] = {}
        for rank, bm in enumerate(sorted(bm25_raw, key=lambda x: x["score"], reverse=True), 1):
            b_ranks[bm["id"]] = rank

        # Xác định mode: hybrid / Qdrant-only / BM25-only
        # (sau khi weekly chunking, IDs giữa Qdrant và BM25 phải match)
        use_bm25_only = False
        use_qdrant_only = False
        if not all_chunk_ids:
            use_bm25_only = True
            logger.info("Hybrid: no results → BM25-only fallback")
        elif v_scores and not b_scores:
            use_qdrant_only = True
            logger.info("Hybrid: BM25 empty → Qdrant-only mode")
        elif b_scores and not v_scores:
            use_bm25_only = True
            logger.info("Hybrid: Qdrant empty → BM25-only mode")
        else:
            overlap = len(set(v_scores) & set(b_scores))
            if overlap == 0:
                # ID mismatch vẫn xảy ra (edge case: cùng week split khác chunk boundaries)
                # → fallback BM25-only vì BM25 có content text đầy đủ
                use_bm25_only = True
                logger.info(f"Hybrid: no ID overlap ({len(v_scores)} Q vs {len(b_scores)} B) → BM25-only fallback")

        # --- Combine results ---
        combined_map: dict[str, dict[str, Any]] = {}

        for rid in all_chunk_ids:
            # Retrieve source_event_ids from whichever source has this ID
            source_ids: list[str] = []
            chunk_text = ""
            chunk_meta = {}

            for r in vector_results:
                if r["id"] == rid:
                    source_ids = r.get("source_event_ids") or []
                    chunk_text = r.get("text", "")
                    chunk_meta = r.get("metadata", {})
                    break

            if not chunk_text:
                for bm in bm25_raw:
                    if bm["id"] == rid:
                        source_ids = bm.get("source_event_ids") or bm.get("metadata", {}).get("source_event_ids") or []
                        chunk_text = bm.get("text", "")
                        chunk_meta = bm.get("metadata", {})
                        break

            if not source_ids:
                source_ids = chunk_meta.get("source_event_ids", [])

            # Per-event filter
            if not _chunk_passes_per_event_filter(source_ids):
                continue

            # Score computation — RRF fusion
            # RRF(k) = 1 / (k + rank) — no normalization needed
            # More robust than alpha-weighted: rank-based, not score-magnitude-based
            if use_bm25_only:
                v_rank = None
                b_rank = b_ranks.get(rid)
            elif use_qdrant_only:
                v_rank = v_ranks.get(rid)
                b_rank = None
            else:
                v_rank = v_ranks.get(rid)
                b_rank = b_ranks.get(rid)

            rrf_score = 0.0
            if v_rank is not None:
                rrf_score += 1.0 / (RRF_K + v_rank)
            if b_rank is not None:
                rrf_score += 1.0 / (RRF_K + b_rank)

            combined_map[rid] = {
                "id": rid,
                "score": round(rrf_score, 4),
                "text": chunk_text,
                "vector_score": round(1.0 / (RRF_K + v_rank), 4) if v_rank is not None else 0.0,
                "bm25_score": round(1.0 / (RRF_K + b_rank), 4) if b_rank is not None else 0.0,
                "metadata": {
                    **chunk_meta,
                    "source_event_ids": source_ids,
                },
            }

        combined = sorted(combined_map.values(), key=lambda x: x["score"], reverse=True)

        # --- Recency boost ---
        if not effective_week_filter and TEMPORAL_ENABLE_RECENCY_BOOST:
            temporal_config = TemporalSearchConfig(
                recency_boost_weight=TEMPORAL_RECENCY_BOOST_WEIGHT,
                enable_recency_boost=True,
                max_boost_factor=TEMPORAL_MAX_BOOST_FACTOR,
            )
            combined = build_recency_boost(combined, temporal_config)

        # --- Cache the combined results (before slicing to top_k) ---
        self._cache.put(
            query=query,
            top_k=top_k,
            alpha=alpha,
            week_filter=effective_week_filter,
            month_filter=effective_month_filter,
            year_filter=effective_year_filter,
            day_filter=effective_day_filter,
            day_of_week_filter=effective_day_of_week_filter,
            results=combined,
        )

        return combined[:top_k], effective_scope_filter, (parsed_query.temporal.type if parsed_query else "general")

    def ask(
        self,
        question: str,
        session_history: list[dict[str, str]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        week_filter: str | None = None,
        day_filter: int | None = None,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """
        Hỏi câu hỏi → retrieval → sinh câu trả lời bằng Gemini 2.5.
        Trả về (answer, results, detected_scope).
        B1: guardrails check trước khi xử lý.
        B4: conversational memory — rolling window.
        """
        if not self._loaded:
            raise RuntimeError("Pipeline chưa ingest dữ liệu. Gọi .ingest() trước.")

        # B1: Guardrails check — fail-open
        guardrails = get_guardrails()
        decision = guardrails.check(question)
        if decision == GuardrailsDecision.BLOCK:
            return (
                guardrails.get_block_message(),
                [],
                "Lịch công tác",
                "blocked",
            )

        # B4: Use session history if provided, otherwise use rolling memory
        effective_history: list[dict[str, str]] | None = None
        if session_history is not None:
            effective_history = session_history
        elif self._session_history:
            effective_history = self._session_history

        # Parse query với QueryParser (LLM-based)
        parsed_query = self._query_parser.parse(question)

        # Resolve week/day filters from parsed query
        resolved_week = week_filter
        resolved_day_of_week: str | None = None
        if parsed_query.temporal.has_reference:
            if not resolved_week:
                intent_for_resolve = TemporalIntent(
                    has_temporal_reference=parsed_query.temporal.has_reference,
                    temporal_type=parsed_query.temporal.type,
                    explicit_year=parsed_query.temporal.year,
                    explicit_month=parsed_query.temporal.month,
                    explicit_day=parsed_query.temporal.day,
                    explicit_day_of_week=parsed_query.temporal.day_of_week,
                    explicit_week=parsed_query.temporal.week,
                )
                resolved_week = resolve_week_filter(intent_for_resolve, self._available_weeks)
            if parsed_query.temporal.day_of_week:
                resolved_day_of_week = parsed_query.temporal.day_of_week

        # Expand query từ chat history để retrieval chính xác hơn
        effective_question = question
        if session_history:
            effective_question = expand_query_from_history(
                question=question,
                chat_history=session_history,
                model_name=self.gemini_model,
            )
            if effective_question != question:
                logger.info(f"Query expanded: '{question}' -> '{effective_question}'")

        results, detected_scope, _temporal_scope_unused = self.hybrid_search(
            query=effective_question,
            top_k=top_k,
            alpha=alpha,
            week_filter=resolved_week,
            day_of_week_filter=resolved_day_of_week,
            original_temporal_intent=None,  # QueryParser đã parse rồi, truyền parsed_query
            parsed_query=parsed_query,
        )
        temporal_scope = parsed_query.temporal.type

        if not results:
            return (
                f"(Đang tìm trong: {detected_scope}) Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn.",
                [],
                detected_scope,
                temporal_scope,
            )

        if session_history:
            answer = generate_answer_with_history(
                question=question,
                retrieval_results=results,
                chat_history=session_history,
                model_name=self.gemini_model,
                scope_context=detected_scope,
                temporal_scope=temporal_scope,
            )
        else:
            answer = generate_answer(
                question=question,
                retrieval_results=results,
                model_name=self.gemini_model,
                scope_context=detected_scope,
                temporal_scope=temporal_scope,
            )

        # B4: Update rolling conversational memory (thread-safe)
        with self._history_lock:
            self._session_history.append({"role": "user", "content": question})
            self._session_history.append({"role": "assistant", "content": answer})
            if len(self._session_history) > self._max_history * 2:
                self._session_history = self._session_history[-self._max_history * 2:]

        return answer, results, detected_scope, temporal_scope

    def get_status(self) -> dict[str, Any]:
        """Trả về trạng thái hiện tại của pipeline."""
        info = self._qdrant.get_collection_info()
        return {
            "loaded": self._loaded,
            "local_chunks": len(self._chunks),          # per-event chunks (JSONL local)
            "bm25_corpus_size": self._bm25.corpus_size,  # per-event BM25 index
            "qdrant_vectors": info.get("vectors_count", info.get("points_count", 0)),  # weekly chunks (Qdrant)
            "qdrant_info": info,
            "available_weeks": self._available_weeks,
            "temporal_recency_boost": TEMPORAL_ENABLE_RECENCY_BOOST,
        }


# --- CLI Entry Points ---
app = typer.Typer(add_completion=False, help="RAG DUE - Hệ thống Q&A Lịch Công Tác")


@app.command()
def ingest(
    jsonl_path: str = typer.Option(..., "--jsonl", "-i", help="Đường dẫn file JSONL"),
    recreate: bool = typer.Option(False, "--recreate", "-r", help="Xoá collection cũ trước khi tạo mới"),
):
    """Ingest dữ liệu JSONL vào vector database."""
    pipeline = RAGPipeline()
    console.print(Panel.fit("[bold cyan]RAG DUE - Ingestion Pipeline[/bold cyan]"))
    try:
        pipeline.ingest(jsonl_path=jsonl_path, recreate_collection=recreate)
        console.print(f"[green]✓ Thành công![/green] Đã ingest từ [yellow]{jsonl_path}[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Lỗi: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q", help="Câu hỏi cần trả lời"),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Số kết quả retrieval"),
    alpha: float = typer.Option(DEFAULT_ALPHA, "--alpha", "-a", help="Tỷ trọng vector search (0-1)"),
):
    """Hỏi câu hỏi và nhận câu trả lời."""
    pipeline = RAGPipeline()

    console.print(Panel.fit(f"[bold cyan]Câu hỏi:[/bold cyan] {question}"))

    try:
        answer, results, _, _ = pipeline.ask(question, top_k=top_k, alpha=alpha)
        console.print(Panel.fit(f"[green]{answer}[/green]", title="Câu trả lời"))

        if results:
            table = Table(title="Nguồn tham chiếu", show_header=True)
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Nội dung", style="white")
            for r in results[:3]:
                table.add_row(f"{r['score']:.3f}", r["text"][:200])
            console.print(table)
    except Exception as e:
        console.print(f"[red]✗ Lỗi: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Kiểm tra trạng thái hệ thống."""
    pipeline = RAGPipeline()
    status_info = pipeline.get_status()

    table = Table(title="RAG DUE System Status", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for k, v in status_info.items():
        table.add_row(str(k), str(v))
    console.print(table)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port chạy API"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host chạy API"),
):
    """Khởi động FastAPI server."""
    console.print(f"[green]Khởi động API server tại http://{host}:{port}[/green]")
    from .api import start_server
    start_server(host=host, port=port)


@app.command()
def ui(
    share: bool = typer.Option(False, "--share", help="Tạo public link Gradio"),
    port: int = typer.Option(7860, "--port", "-p", help="Port chạy Gradio"),
    recreate: bool = typer.Option(False, "--recreate", "-r", help="Xoá collection cũ và re-ingest dữ liệu"),
):
    """Khởi động giao diện Gradio."""
    from .ui import launch_gradio, set_recreate_on_startup
    if recreate:
        set_recreate_on_startup(True)
    print(f"Khoi dong Gradio UI tai http://localhost:{port}")
    launch_gradio(share=share, port=port)


if __name__ == "__main__":
    app()
