"""
BM25 Retriever - Tìm kiếm keyword-based bằng thuật toán BM25.
Kết hợp với vector search để tạo hybrid retrieval.
"""
import logging
from typing import Any

from rank_bm25 import BM25Okapi

try:
    from ..ingestion.chunker import Chunk
except ImportError:
    from ingestion.chunker import Chunk

try:
    from ..config import BM25_TOP_K
except ImportError:
    from config import BM25_TOP_K

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 keyword search trên tập chunks."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: list[Chunk]) -> None:
        """Xây dựng BM25 index từ chunks."""
        import underthesea

        self._chunks = chunks
        self._tokenized_corpus = [
            underthesea.word_tokenize(chunk.text.lower())
            for chunk in chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built: {len(chunks)} chunks, vocab={len(self._bm25.idf):,} terms")

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict[str, Any]]:
        """Tìm kiếm BM25, trả về top_k kết quả với scores."""
        if self._bm25 is None:
            logger.warning("BM25 index chưa được build")
            return []

        import underthesea
        query_tokens = underthesea.word_tokenize(query.lower())
        scores = self._bm25.get_scores(query_tokens)

        scored_results = [
            (i, score, self._chunks[i])
            for i, score in enumerate(scores)
        ]
        scored_results.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_ids = set()
        for idx, score, chunk in scored_results:
            if chunk.id in seen_ids:
                continue
            seen_ids.add(chunk.id)
            results.append({
                "id": chunk.id,
                "score": float(score),
                "text": chunk.text,
                "source_id": chunk.source_id,
                "metadata": chunk.metadata,
            })
            if len(results) >= top_k:
                break

        return results

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Thêm chunks vào index hiện tại."""
        import underthesea

        for chunk in chunks:
            tokens = underthesea.word_tokenize(chunk.text.lower())
            self._chunks.append(chunk)
            self._tokenized_corpus.append(tokens)

        if self._bm25 is not None:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)
