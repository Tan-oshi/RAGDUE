"""
Embedding Module - Tạo vector embeddings từ văn bản tiếng Việt.
Dùng sentence-transformers (intfloat/multilingual-e5-large, dim=1024) chạy local, GPU-aware.
Đồng nhất với upsert_data.ipynb (Colab).
"""
import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

try:
    from ..config import EMBEDDING_MODEL, EMBEDDING_MAX_SEQ_LENGTH
except ImportError:
    from config import EMBEDDING_MODEL, EMBEDDING_MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

_model_cache: SentenceTransformer | None = None


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL,
    device: str | None = None,
) -> SentenceTransformer:
    """Load hoặc trả về cached embedding model."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Tải embedding model: {model_name} trên {device}")
    _model_cache = SentenceTransformer(model_name, device=device)
    _model_cache.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
    logger.info(f"Embedding model loaded. Dim: {_model_cache.get_embedding_dimension()}")
    return _model_cache


def embed_texts(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Tạo embeddings cho danh sách văn bản."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> list[float]:
    """Tạo embedding cho một câu truy vấn."""
    model = get_embedding_model(model_name)
    emb = model.encode(query, normalize_embeddings=True)
    return emb.tolist()


def get_embedding_dimension(model_name: str = EMBEDDING_MODEL) -> int:
    """Lấy chiều embedding của model."""
    model = get_embedding_model(model_name)
    return model.get_embedding_dimension()


def reset_model_cache():
    """Xoá cache model (dùng khi đổi model)."""
    global _model_cache
    _model_cache = None
