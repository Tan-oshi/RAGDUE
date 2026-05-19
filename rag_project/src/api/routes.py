"""
FastAPI Routes - API endpoints cho RAG system.
"""
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path as P
sys.path.insert(0, str(P(__file__).parent.parent.parent))

from src.generation.llm import generate_answer, generate_answer_with_history
from src.retrieval.qdrant_db import QdrantConfig, QdrantManager
from src.retrieval.temporal import resolve_week_filter, TemporalIntent
from src.query_parser import QueryParser

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG DUE - Lịch Công Tác API",
    description="Hệ thống Q&A lịch công tác Trường ĐH Kinh tế, ĐHĐN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global managers
_qdrant: QdrantManager | None = None
_parser: QueryParser | None = None


def get_qdrant() -> QdrantManager:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantManager()
    return _qdrant


def get_parser() -> QueryParser:
    global _parser
    if _parser is None:
        _parser = QueryParser()
    return _parser


# --- Request/Response Models ---
class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = 5
    use_history: bool = True
    week_filter: str | None = None  # override temporal detection


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    session_id: str


class ChatTurn(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[ChatTurn]


# --- In-memory session store ---
_session_store: dict[str, list[dict[str, str]]] = {}


# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "RAG DUE API - Lịch Công Tác Trường ĐH Kinh tế, ĐHĐN"}


@app.get("/health")
def health():
    qdr = get_qdrant()
    info = qdr.get_collection_info()
    return {"status": "ok", "collection": info}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Truy vấn RAG: tìm kiếm context → sinh câu trả lời."""
    qdr = get_qdrant()

    # Temporal detection via QueryParser (LLM-based)
    parser = get_parser()
    parsed = parser.parse(req.question)
    week_filter = req.week_filter

    if not week_filter and parsed.temporal.has_reference:
        available_weeks = qdr.get_available_weeks()
        intent = TemporalIntent(
            has_temporal_reference=parsed.temporal.has_reference,
            temporal_type=parsed.temporal.type,
            explicit_year=parsed.temporal.year,
            explicit_month=parsed.temporal.month,
            explicit_day=parsed.temporal.day,
            explicit_day_of_week=parsed.temporal.day_of_week,
            explicit_week=parsed.temporal.week,
        )
        week_filter = resolve_week_filter(intent, available_weeks)
        logger.info(f"Temporal intent: {parsed.temporal.type} → week_filter={week_filter}")

    retrieval_results = qdr.search(
        query=req.question,
        top_k=req.top_k,
        week_filter=week_filter,
    )

    if not retrieval_results:
        return AskResponse(
            answer="Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong lịch công tác hiện tại.",
            sources=[],
            session_id=req.session_id,
        )

    history = _session_store.get(req.session_id, [])

    if req.use_history and history:
        answer = generate_answer_with_history(
            question=req.question,
            retrieval_results=retrieval_results,
            chat_history=history,
        )
    else:
        answer = generate_answer(
            question=req.question,
            retrieval_results=retrieval_results,
        )

    _session_store.setdefault(req.session_id, []).append(
        {"role": "user", "content": req.question}
    )
    _session_store[req.session_id].append(
        {"role": "assistant", "content": answer}
    )

    sources = [
        {
            "text": r.get("text", "")[:300],
            "score": r.get("score", 0),
            "scope": r.get("scope", ""),
            "week": r.get("week", ""),
        }
        for r in retrieval_results[:3]
    ]

    return AskResponse(answer=answer, sources=sources, session_id=req.session_id)


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    """Lấy lịch sử hội thoại của một session."""
    history = _session_store.get(session_id, [])
    return HistoryResponse(
        session_id=session_id,
        history=[ChatTurn(**t) for t in history],
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Xoá lịch sử hội thoại của một session."""
    if session_id in _session_store:
        del _session_store[session_id]
    return {"message": f"Đã xoá lịch sử session {session_id}"}


@app.get("/sessions")
def list_sessions():
    """Liệt kê các session hiện có."""
    return {
        "sessions": list(_session_store.keys()),
        "total": len(_session_store),
    }


@app.post("/save-history")
def save_history_to_file(path: str = "data/chat_history.json"):
    """Lưu lịch sử hội thoại ra file JSON."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_session_store, f, ensure_ascii=False, indent=2)
    return {"message": f"Đã lưu {len(_session_store)} sessions ra {path}"}


@app.post("/load-history")
def load_history_from_file(path: str = "data/chat_history.json"):
    """Nạp lịch sử hội thoại từ file JSON."""
    load_path = Path(path)
    if not load_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    global _session_store
    _session_store = data
    return {"message": f"Đã nạp {len(_session_store)} sessions từ {path}"}


def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
