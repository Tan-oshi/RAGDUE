"""
Tests cho chunking module.
"""
import pytest
from src.ingestion.chunker import (
    ChunkConfig,
    chunk_records,
    chunk_by_semantic_split,
    _split_sentences,
)


SAMPLE_RECORDS = [
    {
        "id": "test_01",
        "content": "Vào Thứ Hai ngày 04 tháng 05 năm 2026, lúc 08h00 sẽ diễn ra sự kiện: Họp giao ban Trường. Địa điểm tại: Phòng E101. Thành phần tham dự: Theo Quy định. Chủ trì: Hiệu trưởng.",
        "metadata": {"scope": "Lịch làm việc chung", "week": "Tuần 40"},
    },
    {
        "id": "test_02",
        "content": "Vào Thứ Ba ngày 05 tháng 05 năm 2026, lúc 07h00 sẽ diễn ra sự kiện: Hội đồng bảo vệ khóa luận. Địa điểm tại: Phòng E202. Thành phần tham dự: Sinh viên Khóa 48. Chủ trì: Trưởng khoa.",
        "metadata": {"scope": "Lịch làm việc nội bộ", "week": "Tuần 40"},
    },
]


def test_chunk_config_defaults():
    cfg = ChunkConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.separator == "\n"


def test_chunk_config_custom():
    cfg = ChunkConfig(chunk_size=256, chunk_overlap=32)
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32


def test_split_sentences():
    text = "Đây là câu thứ nhất. Đây là câu thứ hai! Đây là câu thứ ba?"
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert "Đây là câu thứ nhất" in sentences[0]


def test_chunk_records_creates_chunks():
    chunks = chunk_records(SAMPLE_RECORDS)
    assert len(chunks) > 0
    assert all(hasattr(c, "id") for c in chunks)
    assert all(hasattr(c, "text") for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_chunk_by_semantic_split():
    chunks = chunk_by_semantic_split(SAMPLE_RECORDS)
    assert len(chunks) > 0
    assert all(hasattr(c, "id") for c in chunks)
    assert all(c.source_id in ("test_01", "test_02") for c in chunks)


def test_chunk_preserves_metadata():
    chunks = chunk_by_semantic_split(SAMPLE_RECORDS)
    for chunk in chunks:
        assert "scope" in chunk.metadata
        assert "week" in chunk.metadata
        assert chunk.metadata["scope"] in ("Lịch làm việc chung", "Lịch làm việc nội bộ")


def test_chunk_ids_unique():
    chunks = chunk_by_semantic_split(SAMPLE_RECORDS)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs phải là duy nhất"


def test_empty_records():
    chunks = chunk_by_semantic_split([])
    assert chunks == []


def test_chunk_text_not_empty():
    chunks = chunk_by_semantic_split(SAMPLE_RECORDS)
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.text.count("  ") < len(chunk.text) / 10
