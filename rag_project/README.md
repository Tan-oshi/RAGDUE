# RAG DUE — Hệ thống Q&A Lịch Công Tác

Hệ thống **RAG (Retrieval-Augmented Generation)** trả lời câu hỏi tự nhiên của sinh viên về lịch công tác hàng tuần của Trường Đại học Kinh tế, Đại học Đà Nẵng.

> Hỏi: *"Tuần này có họp giao ban không, ai chủ trì?"*
>
> Đáp: *"Tuần này có cuộc họp giao ban Phó Hiệu trưởng Nguyễn Văn Minh chủ trì vào Thứ Hai, 14h00 tại Hội trường A."*

## Tính năng

- **Hybrid Search** — Vector search (`intfloat/multilingual-e5-large`, 1024 dim) + BM25, hợp nhất bằng **Reciprocal Rank Fusion (k=60)**
- **Query Parser LLM-based** — Thay regex bằng Gemini 2.5 Flash để parse thời gian, người chủ trì, loại sự kiện từ câu hỏi tiếng Việt tự nhiên
- **Temporal-aware Retrieval** — Recency boost cho tuần gần nhất, hỗ trợ "tuần này", "tháng trước", "năm vừa rồi", ngày cụ thể
- **Multi-turn Chat** — Lưu lịch sử hội thoại theo session, hỗ trợ câu hỏi follow-up
- **Đa giao diện** — Gradio Web UI, FastAPI REST, CLI tương tác
- **Evaluation** — Keyword-overlap metrics (precision, recall, relevance, faithfulness) + LLM-based grading

## Kiến trúc

```
Sinh viên
   │
   ├── Gradio Web UI  :7860
   ├── FastAPI REST   :8000
   └── CLI (typer)
           │
           ┌─────────────── RAG Pipeline ───────────────┐
           │                                            │
     QueryParser (Gemini 2.5 Flash)                   │
           │  parse question → temporal + content       │
           │                                            │
     ┌─────▼──────┐     ┌─────────────────────────┐   │
     │  Retrieval │     │    Generation           │   │
     │            │     │                        │   │
     │ ┌────────┐ │     │  Gemini 2.5 Flash      │   │
     │ │ Qdrant │ │     │  (tiếng Việt, cite)   │   │
     │ └────────┘ │     └────────────────────────┘   │
     │ ┌────────┐ │                                   │
     │ │  BM25  │ │                                   │
     │ └────────┘ │                                   │
     │            │                                   │
     │  RRF Fusion (k=60)                           │
     └────────────┴───────────────────────────────────┘
```

## Cấu trúc thư mục

```
rag_project/
├── data/
│   ├── raw/                    # File JSONL gốc
│   ├── processed/              # Chunks đã xử lý
│   └── chat_history.json       # Lịch sử hội thoại
│
├── notebooks/
│   └── upsert_data.ipynb       # Chạy ingest trên Google Colab
│
├── src/
│   ├── ingestion/              # Nạp & chia nhỏ dữ liệu
│   │   ├── loader.py           # Đọc JSONL, Excel, TXT
│   │   ├── chunker.py          # SentenceSplitter (512 tokens, 90 overlap)
│   │   └── weekly_chunker.py   # Nhóm theo tuần
│   │
│   ├── retrieval/              # Tìm kiếm
│   │   ├── embedding.py        # intfloat/multilingual-e5-large
│   │   ├── qdrant_db.py        # Qdrant Cloud (Cosine, HNSW m=16)
│   │   ├── bm25_retriever.py   # BM25 + underthesea tokenizer
│   │   └── temporal.py         # Recency boost, time filter
│   │
│   ├── query_parser/           # LLM-based query understanding
│   │   ├── schemas.py          # Pydantic: ParsedQuery, TemporalSpec, QueryType
│   │   └── parser.py          # Gemini structured JSON + regex fallback
│   │
│   ├── generation/
│   │   └── llm.py             # Gemini 2.5 Flash, format theo query type
│   │
│   ├── evaluation/
│   │   └── metrics.py        # Keyword overlap + LLM grading
│   │
│   ├── api/
│   │   └── routes.py         # FastAPI: /ask, /history, /status
│   │
│   ├── middleware/
│   │   ├── guardrails.py     # Input validation, PII filter
│   │   └── retry.py         # Exponential backoff (A3 pattern)
│   │
│   ├── main.py               # RAGPipeline orchestration
│   ├── ui.py                # Gradio Web UI
│   ├── config.py            # Toàn bộ cấu hình
│   └── utils/
│       └── error_response.py
│
├── tests/
│   ├── benchmark_queries.json  # 20 test queries theo query type
│   ├── run_benchmark.py      # Chạy benchmark, output JSON metrics
│   ├── analyze_results.py    # So sánh baseline vs post-refactor
│   └── test_*.py            # Unit tests
│
├── .env.example              # Template cấu hình
├── requirements.txt
└── README.md
```

## Cài đặt

### 1. Clone & tạo virtual environment

```bash
cd rag_project
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Cấu hình

```bash
cp .env.example .env
# Mở .env và điền các giá trị cần thiết
```

**`.env.example`:**

```env
# Gemini API (bắt buộc — dùng cho cả generation và query parsing)
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Cloud (đã có default, chỉ thay nếu dùng instance khác)
QDRANT_URL=https://db972175-0a49-4cb0-b451-8ce8b4088e80.eu-central-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=schedule_chunks

# Cấu hình tùy chọn
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=intfloat/multilingual-e5-large
DEFAULT_TOP_K=20
RRF_K=60
```

### 3. Ingest dữ liệu

Dữ liệu JSONL gốc đặt vào `data/raw/`, sau đó chạy:

```bash
# Trên Google Colab (khuyến nghị cho embedding model)
# Xem notebook: notebooks/upsert_data.ipynb

# Hoặc local (cần GPU cho embedding)
python -m src.main ingest --jsonl data/raw/master_lich_tuan.jsonl --recreate
```

### 4. Khởi động

```bash
# Gradio Web UI (khuyến nghị)
python -m src.main ui --port 7860

# FastAPI server
python -m src.main serve --port 8000

# CLI tương tác
python -m src.main ask --question "Lịch họp giao ban tuần này khi nào?"
```

## API

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tuần này có những sự kiện gì?",
    "session_id": "sv001",
    "top_k": 20
  }'
```

### `GET /history/{session_id}`

```bash
curl http://localhost:8000/history/sv001
```

### `GET /status`

```bash
curl http://localhost:8000/status
```

## Ví dụ truy vấn

| Query | Type | Mô tả |
|-------|------|-------|
| `"Lịch làm việc của hiệu trưởng năm vừa rồi"` | year_filter + chairperson | Năm 2025, Hiệu trưởng |
| `"Tuần này có sự kiện gì"` | list + general | Tất cả sự kiện tuần hiện tại |
| `"Tháng 3 có cuộc học gì"` | month_filter | Tháng 3 |
| `"Ai chủ trì họp giao ban"` | who + event | Người chủ trì |
| `"Họp phòng ban thứ 6 do ông nào chủ trì"` | who + event + dow | Thứ 6, người chủ trì |
| `"Có bao nhiêu sự kiện tuần này"` | count | Đếm sự kiện |
| `"Địa điểm tổ chức các sự kiện tuần này"` | where | Địa điểm |
| `"Sự kiện nào vào ngày 15 tháng 5"` | date | Ngày cụ thể |

## Benchmark

```bash
# Chạy benchmark 20 test queries
.venv\Scripts\python.exe tests/run_benchmark.py --output tests/metrics.json

# So sánh baseline vs post-refactor
.venv\Scripts\python.exe tests/analyze_results.py \
    --baseline tests/baseline_metrics.json \
    --current tests/post_refactor_metrics.json
```

## Tham số chính

| Tham số | Giá trị | Mô tả |
|---------|---------|--------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model sinh câu trả lời |
| `QUERY_PARSER_MODEL` | `gemini-2.5-flash` | Model parse câu hỏi |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Embedding model (dim=1024) |
| `RRF_K` | `60` | Hằng số Reciprocal Rank Fusion |
| `DEFAULT_TOP_K` | `20` | Số kết quả retrieval |
| `CHUNK_SIZE` | `512` | Kích thước chunk (tokens) |
| `CHUNK_OVERLAP` | `90` | Overlap giữa các chunk |
| `TEMPORAL_RECENCY_BOOST_WEIGHT` | `0.15` | Hệ số boost recency |
| `QDRANT_COLLECTION` | `schedule_chunks` | Tên collection Qdrant |

## Tech stack

- **Embedding**: `intfloat/multilingual-e5-large` (sentence-transformers)
- **Vector DB**: Qdrant Cloud (Cosine similarity, HNSW)
- **Keyword Search**: BM25 + underthesea tokenizer
- **LLM**: Gemini 2.5 Flash (Google Generative AI)
- **Framework**: LlamaIndex, LangChain
- **UI**: Gradio, FastAPI, Typer (CLI)
- **Evaluation**: RAGAS-like metrics + LLM grading
