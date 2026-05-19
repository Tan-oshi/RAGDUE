# RAG DUE - RAG Pipeline cho Lịch Công Tác ĐH Kinh tế Đà Nẵng

## Chạy dự án

**LUÔN dùng virtual environment `.venv` — KHÔNG cài package toàn cục.**

```bash
cd rag_project
.venv/Scripts/python.exe -m src.main ui
```

Hoặc activate venv trước:
```bash
cd rag_project
.venv/Scripts/activate
python -m src.main ui
```

## Benchmark

```bash
# Baseline (regex-based)
.venv\Scripts\python.exe tests/run_benchmark.py --output tests/baseline_metrics.json

# Post-refactor (LLM-based)
.venv\Scripts\python.exe tests/run_benchmark.py --output tests/post_refactor_metrics.json

# Phân tích
.venv\Scripts\python.exe tests/analyze_results.py --baseline tests/baseline_metrics.json --current tests/post_refactor_metrics.json
```

## Kiến trúc (Refactored)

### QueryParser — LLM-based parsing (`src/query_parser/`)
Thay thế hoàn toàn regex trong `temporal.py`, `content_filter.py`, `query_intent.py`:

- `schemas.py` — Pydantic models: `ParsedQuery`, `TemporalSpec`, `ContentSpec`, `QueryType`
- `parser.py` — `QueryParser` class: Gemini structured JSON parsing, regex fallback

**Tại sao LLM thay vì regex:**
- Regex không detect được: "năm vừa rồi" (= 2025), "tháng trước", "tuần tới", "Phó hiệu trưởng"
- LLM hiểu ngữ cảnh tiếng Việt tự nhiên
- Confidence scoring: 0.95 (LLM success) vs 0.30 (regex fallback)

### Score Fusion — RRF (`src/main.py`)
Thay thế `alpha * v_norm + (1-alpha) * b_norm` bằng **Reciprocal Rank Fusion (k=60)**:

```
RRF_score(chunk) = 1/(60 + rank_vector) + 1/(60 + rank_bm25)
```

**Tại sao RRF:**
- Không cần normalize scores (Qdrant cosine vs BM25 raw)
- Rank-based → robust với outlier scores
- Được chứng minh tốt hơn alpha-weighted trong TREC experiments

### Retrieval (`src/main.py` + `src/retrieval/`)
- **Qdrant**: vector search (multilingual-e5-large, dim=1024, Cosine, HNSW m=16)
- **BM25**: local index trên weekly chunks (underthesea tokenizer)
- **Hybrid**: Qdrant + BM25 chạy song song threads → RRF fusion
- **Fallback**: BM25-only khi Qdrant trả về rỗng, Qdrant-only khi BM25 trả về rỗng

### Chunking (upsert_data.ipynb Colab)
- **Strategy**: nhóm sự kiện theo tuần → aggregate content → SentenceSplitter (chunk_size=512, overlap=90)
- **719 weekly chunks** thay vì 1594 per-event chunks
- **Metadata payload mới**: `week`, `source_event_ids`, `timestamp_unix_min`, `timestamp_unix_max`

### Generation (`src/generation/llm.py`)
- Gemini 2.5 Flash
- `format_context_for_query_type()` — format khác nhau theo query type (count/who/where/list)
- Chat history support

### Evaluation (`src/evaluation/metrics.py`)
- Keyword-overlap metrics (precision, recall, relevance, faithfulness)
- LLM-based grading: `evaluate_with_llm()` — Gemini grades faithfulness và relevance

## Config (`src/config.py`)

| Key | Value | Ghi chú |
|-----|-------|---------|
| `QUERY_PARSER_MODEL` | `gemini-2.5-flash` | Model cho QueryParser |
| `QUERY_PARSER_TEMPERATURE` | `0.0` | Deterministic output |
| `RRF_K` | `60` | RRF constant |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model cho generation |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | dim=1024 |
| `DEFAULT_TOP_K` | `20` | Số kết quả retrieval |

## Cấu trúc file sau refactor

```
src/
  query_parser/
    schemas.py      # Pydantic models (ParsedQuery, TemporalSpec, ContentSpec, QueryType)
    parser.py      # QueryParser (LLM + regex fallback)
    __init__.py
  retrieval/
    qdrant_db.py   # Qdrant vector search
    bm25_retriever.py
    embedding.py
    temporal.py     # Temporal resolve utilities (resolve_week_filter, build_recency_boost, etc.)
                    # NOTE: regex-based detect_temporal_intent() DEPRECATED — QueryParser replaces it
  generation/
    llm.py         # Gemini generation
  evaluation/
    metrics.py     # Keyword + LLM evaluation
  main.py          # RAGPipeline với QueryParser + RRF
  config.py        # QUERY_PARSER_MODEL, RRF_K
```

## Deleted files

- `src/retrieval/content_filter.py` — DEPRECATED: regex content/event matching (QueryParser LLM thay thế)
- `src/retrieval/query_intent.py` — DEPRECATED: cascade intent extraction (QueryParser LLM thay thế)
- `src/retrieval/temporal.py` — DEPRECATED: regex-based `detect_temporal_intent()` đã xóa; giữ lại resolve/boost utilities
