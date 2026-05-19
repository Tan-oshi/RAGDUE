# RAG DUE

Hệ thống **RAG (Retrieval-Augmented Generation)** trả lời câu hỏi tự nhiên về lịch công tác hàng tuần của Trường Đại học Kinh tế, Đại học Đà Nẵng.

## Quick start

```bash
# 1. Cài đặt
cd rag_project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# 2. Cấu hình
cp .env.example .env
# Điền GEMINI_API_KEY vào .env

# 3. Ingest dữ liệu (GPU khuyến nghị cho embedding)
python -m src.main ingest --jsonl data/raw/master_lich_tuan.jsonl --recreate

# 4. Chạy
python -m src.main ui --port 7860   # Gradio Web UI
python -m src.main serve --port 8000 # FastAPI
```

## Tính năng

| Tính năng | Chi tiết |
|---|---|
| **Hybrid Search** | Vector (`intfloat/multilingual-e5-large`, 1024 dim) + BM25, hợp nhất bằng RRF (k=60) |
| **Query Parser** | Gemini 2.5 Flash parse câu hỏi tiếng Việt — thời gian, người chủ trì, loại sự kiện |
| **Temporal Search** | Recency boost, hỗ trợ "tuần này", "tháng trước", "năm vừa rồi", ngày cụ thể |
| **Multi-turn Chat** | Lịch sử hội thoại theo session, hỗ trợ follow-up |
| **Đa giao diện** | Gradio Web UI, FastAPI REST, CLI (typer) |
| **Evaluation** | Keyword-overlap metrics + LLM-based grading |

## Kiến trúc

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/a734fdd2-7109-4684-b5a3-23bcaae81ef0" />


## Cấu trúc dự án

```
rag_project/
├── src/
│   ├── ingestion/          # Nạp dữ liệu & chunking
│   ├── retrieval/          # Vector search, BM25, temporal
│   ├── query_parser/      # LLM-based query understanding
│   ├── generation/        # Gemini generation
│   ├── evaluation/        # RAG metrics
│   ├── api/               # FastAPI endpoints
│   ├── middleware/        # Guardrails, retry
│   ├── main.py            # RAGPipeline
│   ├── ui.py              # Gradio UI
│   └── config.py          # Cấu hình
├── tests/
│   ├── benchmark_queries.json
│   ├── run_benchmark.py
│   └── test_*.py
├── data/                  # JSONL gốc, chunks, vector store
├── notebooks/             # Colab notebooks
├── .env.example
├── requirements.txt
└── README.md
```

## API

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tuần này có sự kiện gì?", "session_id": "sv001"}'
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

| Query | Loại |
|---|---|
| Tuần này có sự kiện gì | list + general |
| Tháng 3 có cuộc học gì | month_filter |
| Ai chủ trì họp giao ban | who + event |
| Họp phòng ban thứ 6 do ông nào chủ trì | who + event + dow |
| Có bao nhiêu sự kiện tuần này | count |
| Địa điểm tổ chức các sự kiện tuần này | where |
| Sự kiện nào vào ngày 15 tháng 5 | date |
| Phó hiệu trưởng có lịch họp gì tuần tới | chairperson + week |

## Cấu hình

Các tham số chính trong [src/config.py](src/config.py):

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model sinh câu trả lời |
| `QUERY_PARSER_MODEL` | `gemini-2.5-flash` | Model parse câu hỏi |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Embedding (dim=1024) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `DEFAULT_TOP_K` | `20` | Số kết quả retrieval |
| `CHUNK_SIZE` | `512` | Kích thước chunk (tokens) |
| `CHUNK_OVERLAP` | `90` | Overlap giữa các chunk |
| `QDRANT_COLLECTION` | `schedule_chunks` | Tên collection Qdrant |


## Tech stack

**Embedding**: `intfloat/multilingual-e5-large` · **Vector DB**: Qdrant Cloud · **Keyword**: BM25 + underthesea · **LLM**: Gemini 2.5 Flash · **Framework**: LlamaIndex, LangChain · **UI**: Gradio, FastAPI, Typer
