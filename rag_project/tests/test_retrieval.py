"""
Tests cho retrieval module.
"""
import pytest
from src.ingestion.chunker import chunk_by_semantic_split, Chunk
from src.retrieval.bm25_retriever import BM25Retriever


SAMPLE_CHUNKS = [
    Chunk(
        id="chunk_01",
        text="Thứ Hai ngày 04 tháng 05 năm 2026, lúc 08h00 Họp giao ban Trường tại Phòng E101. Thành phần: Theo Quy định. Chủ trì: Hiệu trưởng.",
        metadata={"scope": "chung", "week": "40"},
        source_id="test_01",
    ),
    Chunk(
        id="chunk_02",
        text="Thứ Ba ngày 05 tháng 05 năm 2026, lúc 07h00 Hội đồng bảo vệ khóa luận tại Phòng E202. Sinh viên Khóa 48.",
        metadata={"scope": "nội bộ", "week": "40"},
        source_id="test_02",
    ),
    Chunk(
        id="chunk_03",
        text="Thứ Tư ngày 06 tháng 05 năm 2026, lúc 14h00 Quỹ Nafosted kiểm tra kinh phí đề tài tại Phòng H101.",
        metadata={"scope": "chung", "week": "40"},
        source_id="test_03",
    ),
]


def test_bm25_build_index():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS)
    assert retriever.corpus_size == 3


def test_bm25_search_returns_results():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS)
    results = retriever.search(query="Họp giao ban", top_k=2)
    assert len(results) <= 2
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)


def test_bm25_search_scores_positive():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS)
    results = retriever.search(query="Nafosted", top_k=3)
    if results:
        assert all(r["score"] >= 0 for r in results)


def test_bm25_search_deduplicates():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS)
    results = retriever.search(query="Thứ", top_k=10)
    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), "Kết quả không được trùng lặp"


def test_bm25_empty_query():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS)
    results = retriever.search(query="", top_k=5)
    assert isinstance(results, list)


def test_bm25_no_index():
    retriever = BM25Retriever()
    retriever._bm25 = None
    results = retriever.search(query="test", top_k=5)
    assert results == []


def test_bm25_add_chunks():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS[:1])
    assert retriever.corpus_size == 1
    retriever.add_chunks(SAMPLE_CHUNKS[1:])
    assert retriever.corpus_size == 3


def test_bm25_chunk_metadata_preserved():
    retriever = BM25Retriever(chunks=SAMPLE_CHUNKS)
    results = retriever.search(query="khóa luận", top_k=2)
    for r in results:
        assert "source_id" in r
        assert "metadata" in r
