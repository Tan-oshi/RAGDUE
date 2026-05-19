# RAG DUE - Hệ thống Q&A Lịch Công Tác

Hệ thống **RAG (Retrieval-Augmented Generation)** trả lời câu hỏi của sinh viên về lịch công tác hàng tuần của Trường Đại học Kinh tế, Đại học Đà Nẵng.

## Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────┐
│                        User (Sinh viên)                       │
└─────────────────────────────┬────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐         ┌─────▼─────┐        ┌────▼────┐
    │ Gradio  │         │  FastAPI  │        │   CLI   │
    │   Web   │         │ REST API  │        │  typer  │
    └────┬────┘         └─────┬─────┘        └────┬────┘
         └─────────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   RAG Pipeline      │
                    │  ┌───────────────┐  │
                    │  │  Generation   │  │
                    │  │  Gemini 2.5   │  │
                    │  └───────────────┘  │
                    │  ┌───────────────┐  │
                    │  │   Retrieval   │  │
                    │  │ Vector+BM25   │  │
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
  ┌──────▼──────┐      ┌──────▼────── ┐     ┌──────▼──────┐
  │  Qdrant     │      │   BM25       │     │  Embedding  │
  │  (Vector)   │      │  (Keyword)   │     │  (bge-m3)   │
  └─────────────┘      └──────────────┘     └─────────────┘
```

## Cấu trúc thư mục

```
rag_project/
├── data/
│   ├── raw/                    # Dữ liệu gốc JSONL
│   ├── processed/              # Chunks đã xử lý
│   └── vector_storage/         # Qdrant local storage
│
├── notebooks/
│   └── eda_and_test.ipynb      # EDA & thử nghiệm chunk size
│
├── src/
│   ├── ingestion/              # Data Loading & Chunking
│   │   ├── loader.py          # Đọc JSONL, Excel, TXT
│   │   └── chunker.py         # Chia nhỏ văn bản (LlamaIndex)
│   │
│   ├── retrieval/             # Vector Search
│   │   ├── embedding.py       # BAAI/bge-m3 embeddings
│   │   ├── qdrant_db.py       # Qdrant client
│   │   └── bm25_retriever.py  # BM25 keyword search
│   │
│   ├── generation/            # Text Generation
│   │   ├── prompts.py         # Prompt templates tiếng Việt
│   │   └── llm.py             # Gemini 2.5 API integration
│   │
│   ├── evaluation/           # Đánh giá RAG
│   │   └── metrics.py        # RAGAS-like scores
│   │
│   ├── api/
│   │   └── routes.py         # FastAPI endpoints
│   │
│   ├── main.py               # CLI + Pipeline orchestration
│   └── ui.py                 # Gradio Web UI
│
├── tests/
│   ├── test_chunker.py
│   └── test_retrieval.py
│
├── .env                       # API keys (KHÔNG commit)
├── requirements.txt
└── README.md
```

## Cài đặt

### 1. Cài đặt dependencies

```bash
cd rag_project
pip install -r requirements.txt
```

### 2. Cấu hình

Tạo file `.env` (tham khảo `.env.example`):

```env
# Bắt buộc - Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Vector DB (local)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=lich_tuan

# Embedding model
EMBEDDING_MODEL=BAAI/bge-m3
```

### 3. Khởi động Qdrant (local)

```bash
# Docker
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/data/vector_storage:/qdrant/storage \
    qdrant/qdrant

# Hoặc download binary từ https://qdrant.tech/
```

### 4. Ingest dữ liệu

```bash
# Di chuyển file JSONL vào thư mục raw
cp /path/to/master_lich_tuan.jsonl data/raw/

# Chạy ingestion
python -m src.main ingest --jsonl data/raw/master_lich_tuan.jsonl --recreate
```

### 5. Khởi động

```bash
# Gradio Web UI (khuyến nghị)
python -m src.main ui --port 7860

# Hoặc FastAPI server
python -m src.main serve --port 8000

# Hoặc CLI tương tác
python -m src.main ask --question "Lịch họp giao ban tuần này khi nào?"
```

## Cách sử dụng

### Gradio Web UI
1. Mở trình duyệt: `http://localhost:7860`
2. Nhập câu hỏi bằng tiếng Việt
3. Xem câu trả lời + nguồn tham chiếu

### CLI
```bash
# Hỏi câu hỏi
python -m src.main ask -q "Tuần này có những sự kiện gì?"

# Kiểm tra trạng thái
python -m src.main status
```

### API
```bash
# Hỏi câu hỏi
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Lịch họp giao ban", "session_id": "student001"}'

# Lấy lịch sử hội thoại
curl http://localhost:8000/history/student001
```

## Tính năng chính

| Tính năng | Mô tả |
|-----------|--------|
| **Hybrid Search** | Kết hợp vector (BAAI/bge-m3) + BM25, alpha=0.6 |
| **Multi-turn Chat** | Lưu history theo session, hỗ trợ follow-up questions |
| **Vietnamese RAG** | Prompt + chunking tối ưu cho tiếng Việt |
| **Gradio UI** | Giao diện chat thân thiện, hiển thị nguồn |
| **FastAPI** | REST API cho tích hợp hệ thống khác |
| **Evaluation** | RAGAS-like metrics: precision, recall, relevance, faithfulness |
| **Hybrid alpha tuning** | Slider điều chỉnh tỷ trọng vector/keyword |

## Lưu ý

- **Offline**: Embedding model chạy local (BAAI/bge-m3), không cần internet để retrieval
- **Online**: Gemini API cần internet để sinh câu trả lời
- **Qdrant**: Cần chạy local hoặc dùng Qdrant Cloud
- **Chat History**: Được lưu trong memory, export được ra JSON

## License

MIT
