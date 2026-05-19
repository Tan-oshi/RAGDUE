"""
Weekly Chunker - Nhóm per-event records theo tuần, aggregate content, và chia chunk.
Đồng nhất với upsert_data.ipynb (cell 6): group by week → join content → SentenceSplitter.
Mục đích: tạo weekly chunks local để BM25 đồng nhất với Qdrant collection.
"""
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SentenceSplitter

from .chunker import Chunk

try:
    from ..config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATOR
except ImportError:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATOR

logger = logging.getLogger(__name__)


@dataclass
class WeeklyChunk(Chunk):
    """Weekly chunk với danh sách source event IDs."""
    source_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["source_event_ids"] = self.source_event_ids
        return base


def load_and_group_by_week(jsonl_path: str) -> dict[str, dict[str, Any]]:
    """
    Đọc JSONL, nhóm records theo tuần.
    Trả về dict: week_key → {
        "content": concatenated_content,
        "source_event_ids": [id1, id2, ...],
        "timestamp_unix_min": int,
        "timestamp_unix_max": int,
        "records": [original_record_dict, ...],
    }
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {jsonl_path}")

    weekly_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "content": "",
        "source_event_ids": [],
        "timestamp_unix_min": None,
        "timestamp_unix_max": None,
        "records": [],
    })

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta = record.get("metadata", {}) or {}

            # Extract week from metadata
            week = meta.get("week", "unknown").strip()
            if not week:
                week = "unknown"

            group = weekly_groups[week]
            group["content"] += " " + record.get("content", "")
            group["source_event_ids"].append(record.get("id", ""))
            group["records"].append(record)

            # Update timestamp range (metadata.timestamp is in seconds)
            ts = meta.get("timestamp")
            if ts is not None:
                ts_ms = int(ts) * 1000
                if group["timestamp_unix_min"] is None or ts_ms < group["timestamp_unix_min"]:
                    group["timestamp_unix_min"] = ts_ms
                if group["timestamp_unix_max"] is None or ts_ms > group["timestamp_unix_max"]:
                    group["timestamp_unix_max"] = ts_ms

    # Trim whitespace from concatenated content
    for group in weekly_groups.values():
        group["content"] = group["content"].strip()

    logger.info(f"Grouped {sum(len(g['records']) for g in weekly_groups.values())} events into {len(weekly_groups)} weeks")
    return dict(weekly_groups)


def chunk_weekly_groups(
    weekly_groups: dict[str, dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[WeeklyChunk]:
    """
    Chia weekly groups thành chunks sử dụng SentenceSplitter.
    Đồng nhất với upsert_data.ipynb cell 6.
    Mỗi weekly group → nhiều chunks, mỗi chunk giữ nguyên week metadata + source_event_ids.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )

    chunks: list[WeeklyChunk] = []

    # Sort weeks consistently for reproducible chunk IDs
    sorted_weeks = sorted(weekly_groups.keys(), key=_week_sort_key)

    for week_key in sorted_weeks:
        group = weekly_groups[week_key]
        combined_content = group["content"]
        source_ids = group["source_event_ids"]
        ts_min = group["timestamp_unix_min"]
        ts_max = group["timestamp_unix_max"]

        if not combined_content:
            continue

        # Split the weekly content
        chunk_texts = splitter.split_text(combined_content)

        for i, chunk_text in enumerate(chunk_texts):
            chunk_id_str = f"week_{week_key.replace(' ', '_')}_chunk_{i}"
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id_str))

            chunks.append(WeeklyChunk(
                id=point_uuid,
                text=chunk_text,
                metadata={
                    "week": week_key,
                    "source_event_ids": source_ids,
                    "timestamp_unix_min": ts_min,
                    "timestamp_unix_max": ts_max,
                },
                source_id=week_key,
                chunk_index=i,
                source_event_ids=source_ids,
            ))

    logger.info(f"Weekly chunking: {len(weekly_groups)} weeks → {len(chunks)} weekly chunks")
    return chunks


def build_weekly_chunks(jsonl_path: str) -> list[WeeklyChunk]:
    """
    Load JSONL → group by week → chunk → return weekly chunks.
    Kết quả đồng nhất với dữ liệu trong Qdrant collection (719 chunks).
    """
    weekly_groups = load_and_group_by_week(jsonl_path)
    chunks = chunk_weekly_groups(weekly_groups)
    logger.info(f"Built {len(chunks)} weekly chunks from {jsonl_path}")
    return chunks


def get_per_event_metadata(
    jsonl_path: str,
    event_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Load per-event metadata từ JSONL cho các event IDs.
    Dùng để post-filter weekly chunks bằng per-event fields (scope, month, year, etc.).
    """
    path = Path(jsonl_path)
    if not path.exists():
        return {}

    # Build index: id → record
    event_index: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            event_index[record.get("id", "")] = record

    result = {}
    for eid in event_ids:
        if eid in event_index:
            result[eid] = event_index[eid]
    return result


def _week_sort_key(week_str: str) -> tuple:
    """Sort key cho week string: extract số, sort theo số."""
    import re
    m = re.search(r"W?(\d+)", week_str, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, week_str)
