"""
Chunking Module - Chia nhỏ văn bản thành các đoạn có ngữ cảnh.
Sử dụng LlamaIndex SimpleNodeParser làm core, bổ sung tự định nghĩa.
"""
import json
import re
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document, Node

logger = logging.getLogger(__name__)


try:
    from ...config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATOR, SEMANTIC_MIN_CHUNK, SEMANTIC_MAX_CHUNK
except ImportError:
    try:
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATOR, SEMANTIC_MIN_CHUNK, SEMANTIC_MAX_CHUNK
    except ImportError:
        CHUNK_SIZE = 512
        CHUNK_OVERLAP = 90
        CHUNK_SEPARATOR = "\n"
        SEMANTIC_MIN_CHUNK = 128
        SEMANTIC_MAX_CHUNK = 1024


@dataclass
class ChunkConfig:
    """Cấu hình chunking cho LlamaIndex."""
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    separator: str = CHUNK_SEPARATOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator,
        }


@dataclass
class Chunk:
    """Đại diện một chunk sau khi chia nhỏ."""
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
        }


def chunk_records(
    records: list[dict[str, Any]],
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """
    Chia nhỏ các bản ghi thành chunks sử dụng LlamaIndex SimpleNodeParser.
    Mỗi bản ghi JSONL = một Document, giữ nguyên metadata.
    """
    if config is None:
        config = ChunkConfig()

    docs = _records_to_documents(records)
    parser = SimpleNodeParser(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separator=config.separator,
    )
    nodes = parser.get_nodes_from_documents(docs)

    chunks = []
    for node in nodes:
        chunks.append(Chunk(
            id=node.id_,
            text=node.text,
            metadata=node.metadata,
            source_id=node.metadata.get("source_id", ""),
            chunk_index=node.metadata.get("chunk_index", 0),
        ))

    logger.info(f"Đã chia thành {len(chunks)} chunks từ {len(records)} bản ghi")
    return chunks


def chunk_by_semantic_split(
    records: list[dict[str, Any]],
    min_chunk_size: int = SEMANTIC_MIN_CHUNK,
    max_chunk_size: int = SEMANTIC_MAX_CHUNK,
) -> list[Chunk]:
    """
    Chia nhỏ theo ngữ nghĩa - tách tại ranh giới câu/tự động.
    Phù hợp cho dữ liệu lịch có cấu trúc: [Ngày | Giờ | Sự kiện | Địa điểm | Thành phần | Chủ trì].
    """
    # Import here to avoid circular import at module level
    from .loader import extract_date_metadata

    chunks = []
    for record in records:
        content = record.get("content", "")
        meta = record.get("metadata", {})
        # Extract full date fields (day, day_month, day_year, day_of_week) from content
        date_fields = extract_date_metadata(content)
        meta = {**date_fields, **meta}  # meta (event_name, scope...) wins over date_fields

        sentences = _split_sentences(content)
        current_chunk_texts = []
        current_size = 0

        for sent in sentences:
            sent_size = len(sent)
            if current_size + sent_size > max_chunk_size and current_chunk_texts:
                chunk_text = " ".join(current_chunk_texts)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(Chunk(
                        id=f"{record['id']}_chunk_{len(chunks)}",
                        text=chunk_text,
                        metadata={**meta, "source_id": record["id"]},
                        source_id=record["id"],
                        chunk_index=len(chunks),
                    ))
                current_chunk_texts = []
                current_size = 0
            current_chunk_texts.append(sent)
            current_size += sent_size

        if current_chunk_texts:
            chunk_text = " ".join(current_chunk_texts)
            chunks.append(Chunk(
                id=f"{record['id']}_chunk_{len(chunks)}",
                text=chunk_text,
                metadata={**meta, "source_id": record["id"]},
                source_id=record["id"],
                chunk_index=len(chunks),
            ))

    logger.info(f"Semantic split: {len(records)} bản ghi → {len(chunks)} chunks")
    return chunks


def _records_to_documents(records: list[dict[str, Any]]) -> list[Document]:
    """Chuyển JSONL records thành LlamaIndex Documents."""
    docs = []
    for record in records:
        doc = Document(
            text=record.get("content", ""),
            metadata={
                "source_id": record.get("id", ""),
                **{k: v for k, v in record.get("metadata", {}).items()},
            },
            id_=record.get("id", ""),
        )
        docs.append(doc)
    return docs


def load_records_direct(file_path: str) -> list[Chunk]:
    """
    Đọc JSONL trực tiếp, mỗi record = 1 chunk.
    KHÔNG enrich metadata với date fields — giữ nguyên event_name từ JSONL.
    Giống cách tiếp cận trong notebook Colab: mỗi bản ghi nhỏ (~356 chars),
    chunking thêm sẽ mất ngữ cảnh.

    Cải tiến từ upsert_data.ipynb:
    - Dùng UUID v5 (URL namespace) làm point ID thay vì hash
    - Thêm timestamp_unix (milliseconds) cho Qdrant range filter
    """
    chunks: list[Chunk] = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id", f"chunk_{i}")

            # UUID v5 (URL namespace) - stable, collision-resistant
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))

            # Enrich metadata: add timestamp_unix (ms) from timestamp (seconds)
            meta = {**record.get("metadata", {})}
            if "timestamp" in meta and meta["timestamp"] is not None:
                meta["timestamp_unix"] = int(meta["timestamp"]) * 1000

            chunks.append(Chunk(
                id=point_uuid,       # UUID v5 dùng làm Qdrant point ID
                text=record.get("content", ""),
                metadata={
                    **meta,
                    "source_id": record_id,
                },
                source_id=record_id,
                chunk_index=i,
            ))

    logger.info(f"Direct load: {len(chunks)} chunks from {file_path}")
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Tách câu tiếng Việt, giữ dấu câu làm ranh giới."""
    sentences = re.split(r"(?<=[.!?。；])+\s*", text)
    return [s.strip() for s in sentences if s.strip()]


_DATE_RE = re.compile(
    r"ngày\s+\d{1,2}\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)


def _extract_date_fields(content: str) -> dict[str, Any]:
    """
    Trích xuất month và year từ nội dung văn bản.
    Hỗ trợ:
      - Format đầy đủ: 'Thứ Hai ngày 04 tháng 05 năm 2026' → month=5, year=2026
      - Format ngắn: 'Vào HAI – 05/01' → month=1, year inferred (Aug-Dec 2025, Jan-Jul 2026)
    """
    m = _DATE_RE.search(content)
    if m:
        return {"month": int(m.group(1)), "year": int(m.group(2))}

    # Format ngắn dd/mm: "05/01" hoặc "5/1"
    short_re = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})\b")
    for sm in short_re.finditer(content):
        day, month = int(sm.group(1)), int(sm.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            # Infer year: tháng 8-12 → 2025, tháng 1-7 → 2026 (năm học Aug 2025 - May 2026)
            year = 2026 if month <= 7 else 2025
            return {"month": month, "year": year}
    return {}
