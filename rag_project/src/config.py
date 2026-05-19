"""
Config - Tất cả tham số cấu hình hệ thống RAG DUE.
Import: from src.config import cfg
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")
VECTOR_STORAGE_PATH = Path("data/vector_storage")
CHAT_HISTORY_PATH = Path("data/chat_history.json")


# ---------------------------------------------------------------------------
# Ingestion / Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 90
CHUNK_SEPARATOR = "\n"
SEMANTIC_MIN_CHUNK = 128
SEMANTIC_MAX_CHUNK = 1024


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_MAX_SEQ_LENGTH = 1024
EMBEDDING_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
BM25_TOKENIZER = "underthesea"
BM25_TOP_K = 5


# ---------------------------------------------------------------------------
# Qdrant Cloud
# ---------------------------------------------------------------------------
QDRANT_URL = "https://db972175-0a49-4cb0-b451-8ce8b4088e80.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzYwM2QxOTMtOGZiNy00MTI4LWE3YWEtMWJkYmRlY2M0MjhkIn0.1BZW96fXKU-BtQyeGR6tfz41BhKQAFUUh2MA1ccRgDU"
QDRANT_COLLECTION = "schedule_chunks"
QDRANT_DISTANCE = "Cosine"
QDRANT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.6          # tỷ trọng vector (1.0=vector, 0.0=BM25)
DEFAULT_TOP_K = 20

# ---------------------------------------------------------------------------
# Temporal Search
# ---------------------------------------------------------------------------
TEMPORAL_RECENCY_BOOST_WEIGHT = 0.15  # hệ số boost cho tuần gần nhất khi không đề cập thời gian
TEMPORAL_ENABLE_RECENCY_BOOST = True
TEMPORAL_MAX_BOOST_FACTOR = 2.0       # multiplier tối đa cho tuần mới nhất
DEFAULT_TEMPORAL_FILTER: str | None = None  # filter mặc định theo tuần (None = auto/recency)


# ---------------------------------------------------------------------------
# LLM / Generation
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_TOKENS = 8192
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"


# ---------------------------------------------------------------------------
# Query Parser (LLM-based)
# ---------------------------------------------------------------------------
QUERY_PARSER_MODEL = "gemini-2.5-flash"
QUERY_PARSER_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Retrieval Fusion
# ---------------------------------------------------------------------------
RRF_K = 60  # Reciprocal Rank Fusion constant


# ---------------------------------------------------------------------------
# API / UI
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
GRADIO_PORT = 7860


# ---------------------------------------------------------------------------
# Query Cache
# ---------------------------------------------------------------------------
QUERY_CACHE_TTL_SECONDS = 3600
QUERY_CACHE_MAX_SIZE = 500


# ---------------------------------------------------------------------------
# Retry (A3 pattern)
# ---------------------------------------------------------------------------
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", 3))
RETRY_INITIAL_DELAY = float(os.getenv("RETRY_INITIAL_DELAY", 0.5))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", 2.0))


# ---------------------------------------------------------------------------
# Query Abbreviations (normalize before parsing)
# ---------------------------------------------------------------------------
QUERY_ABBREVIATIONS: dict[str, str] = {
    "ht": "học tập",
    "ct": "công tác",
    "qd": "quy định",
    "pt": "phòng",
    "cb": "cán bộ",
    "nv": "nhân viên",
    "sv": "sinh viên",
}


# ---------------------------------------------------------------------------
# Dataclass-based config (dùng cho RAGPipeline / QdrantManager)
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """Cấu hình chunking — dùng với ChunkConfig."""
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    separator: str = CHUNK_SEPARATOR


@dataclass
class QdrantSettings:
    """Cấu hình Qdrant Cloud — dùng với QdrantConfig."""
    url: str = QDRANT_URL
    api_key: str = QDRANT_API_KEY
    collection_name: str = QDRANT_COLLECTION
    distance: str = QDRANT_DISTANCE


@dataclass
class EmbeddingSettings:
    """Cấu hình embedding — intfloat/multilingual-e5-large (dim=1024)."""
    model_name: str = EMBEDDING_MODEL
    max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH
    batch_size: int = EMBEDDING_BATCH_SIZE


@dataclass
class GenerationSettings:
    """Cấu hình LLM generation."""
    model_name: str = GEMINI_MODEL
    temperature: float = GEMINI_TEMPERATURE
    max_output_tokens: int = GEMINI_MAX_TOKENS


@dataclass
class SearchSettings:
    """Cấu hình hybrid search."""
    default_top_k: int = DEFAULT_TOP_K
    default_alpha: float = DEFAULT_ALPHA
    temporal_recency_boost_weight: float = TEMPORAL_RECENCY_BOOST_WEIGHT
    temporal_enable_recency_boost: bool = TEMPORAL_ENABLE_RECENCY_BOOST
    temporal_max_boost_factor: float = TEMPORAL_MAX_BOOST_FACTOR
    default_temporal_filter: str | None = DEFAULT_TEMPORAL_FILTER


@dataclass
class AppSettings:
    """Cấu hình ứng dụng / UI."""
    api_host: str = API_HOST
    api_port: int = API_PORT
    gradio_port: int = GRADIO_PORT
    data_raw_path: Path = field(default_factory=lambda: DATA_RAW_PATH)
    data_processed_path: Path = field(default_factory=lambda: DATA_PROCESSED_PATH)
    vector_storage_path: Path = field(default_factory=lambda: VECTOR_STORAGE_PATH)
    chat_history_path: Path = field(default_factory=lambda: CHAT_HISTORY_PATH)


@dataclass
class Config:
    """Gộp toàn bộ cấu hình — dùng: cfg.chunking.chunk_size"""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    app: AppSettings = field(default_factory=AppSettings)


cfg = Config()
