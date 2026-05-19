"""
Evaluation Module - Đánh giá chất lượng RAG pipeline bằng RAGAS.
Thay thế hoàn toàn keyword-overlap bằng LLM-based metrics.
RAGAS 0.4.x API: evaluate() + old-underscore metrics (_faithfulness, etc.).
"""
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAGAS LLM / Embedding setup (lazy singleton)
# ---------------------------------------------------------------------------

_ragas_llm: Optional[Any] = None
_ragas_embeddings: Optional[Any] = None


def _get_ragas_llm(
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> Any:
    """Tạo/sử dụng singleton InstructorLLM cho RAGAS metrics."""
    global _ragas_llm
    if _ragas_llm is not None:
        return _ragas_llm

    try:
        from ragas.llms import llm_factory
        from instructor import from_genai
        from google import genai
    except ImportError as e:
        logger.warning(f"RAGAS LLM deps missing: {e}")
        return None

    if api_key is None:
        from dotenv import load_dotenv
        # Try both: cwd is project root, and cwd is rag_project
        loaded = load_dotenv("rag_project/.env")
        if not loaded:
            load_dotenv(".env")
        api_key = os.getenv("GEMINI_API_KEY", "")

    try:
        client = genai.Client(api_key=api_key)
        inst = from_genai(client)
        _ragas_llm = llm_factory(model, provider="gemini", client=inst)
        logger.info(f"RAGAS LLM initialized: {model}")
    except Exception as e:
        logger.warning(f"Failed to initialize RAGAS LLM: {e}")
        return None

    return _ragas_llm


def _get_ragas_embeddings(
    model: str = "intfloat/multilingual-e5-large",
) -> Any:
    """Tạo embedding model cho AnswerRelevancy metric.

    Sử dụng LangChain HuggingFaceEmbeddings với sentence-transformers.
    Fallback: trả về None nếu không có embedding model.
    """
    global _ragas_embeddings
    if _ragas_embeddings is not None:
        return _ragas_embeddings

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        _ragas_embeddings = HuggingFaceEmbeddings(
            model_name=model,
            cache_folder=None,
        )
        logger.info(f"RAGAS Embeddings initialized: {model}")
        return _ragas_embeddings
    except ImportError:
        logger.warning("langchain_community not available for embeddings")
    except Exception as e:
        logger.warning(f"Failed to initialize RAGAS embeddings: {e}")

    return None


def reset_ragas_client() -> None:
    """Reset singleton — dùng khi cần reinitialize với config mới."""
    global _ragas_llm, _ragas_embeddings
    _ragas_llm = None
    _ragas_embeddings = None


# ---------------------------------------------------------------------------
# Metric factories (lazy-init với retry support)
# ---------------------------------------------------------------------------

def _build_metrics(
    llm: Any,
    embeddings: Any,
    raise_on_error: bool = False,
) -> list[Any]:
    """Khởi tạo 2 RAGAS metrics: Faithfulness, AnswerRelevancy.

    Drop ContextPrecision + ContextRecall vì cần ground_truth (luôn NaN trên dataset hiện tại).
    Giảm LLM calls từ 4 metrics → 2 metrics: ~140-160 → ~3-5 calls cho full benchmark 20 queries.
    """
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy

    metrics = []
    errors = []

    try:
        f = Faithfulness(llm=llm)
        metrics.append(f)
    except Exception as e:
        msg = f"Faithfulness init failed: {e}"
        errors.append(msg)
        logger.warning(msg)

    try:
        a = AnswerRelevancy(llm=llm, embeddings=embeddings)
        metrics.append(a)
    except Exception as e:
        msg = f"AnswerRelevancy init failed: {e}"
        errors.append(msg)
        logger.warning(msg)

    if raise_on_error and not metrics:
        raise RuntimeError(f"All metrics failed to init: {errors}")

    return metrics


# ---------------------------------------------------------------------------
# Evaluation Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RAGEvaluationResult:
    """Kết quả đánh giá một lượt RAG."""
    question: str
    answer: str
    retrieved_contexts: list[str]
    ground_truth: Optional[str] = None

    # RAGAS metric scores (0.0-1.0, NaN nếu không tính được)
    faithfulness: float = float("nan")
    answer_relevancy: float = float("nan")

    # Legacy aliases (để tương thích)
    answer_faithfulness: float = float("nan")
    answer_relevance: float = float("nan")
    ragas_score: float = float("nan")

    # Error info
    eval_error: Optional[str] = None

    def __post_init__(self):
        # Sync legacy aliases
        self.answer_faithfulness = self.faithfulness
        self.answer_relevance = self.answer_relevancy

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer_preview": self.answer[:200],
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "ragas_score": self.ragas_score,
            "eval_error": self.eval_error,
        }


@dataclass
class DatasetEvaluationResult:
    """Kết quả đánh giá dataset."""
    results: list[RAGEvaluationResult] = field(default_factory=list)
    num_questions: int = 0

    avg_faithfulness: float = float("nan")
    avg_answer_relevancy: float = float("nan")
    avg_ragas_score: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_questions": self.num_questions,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_answer_relevancy": self.avg_answer_relevancy,
            "avg_ragas_score": self.avg_ragas_score,
        }


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_rag_response(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    ground_truth: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> RAGEvaluationResult:
    """
    Đánh giá một RAG response bằng RAGAS.
    Dùng LLM-based metrics: Faithfulness, AnswerRelevancy (2 metrics — giảm từ 4 để tối ưu quota).

    Retry logic: Nếu API quota exceeded → trả về NaN scores + error message.
    """
    result = RAGEvaluationResult(
        question=question,
        answer=answer,
        retrieved_contexts=[c.get("text", "") for c in retrieved_chunks],
        ground_truth=ground_truth,
    )

    if not retrieved_chunks or not answer:
        result.eval_error = "Empty context or answer"
        return result

    # Lazy-init LLM và embeddings
    llm = _get_ragas_llm(model=model, api_key=api_key)
    embeddings = _get_ragas_embeddings()

    if llm is None:
        result.eval_error = "RAGAS LLM not available"
        return result

    metrics = _build_metrics(llm, embeddings, raise_on_error=False)
    if not metrics:
        result.eval_error = "No metrics initialized"
        return result

    try:
        from ragas.dataset_schema import EvaluationDataset
        from ragas import evaluate

        contexts_str = [c.get("text", "") for c in retrieved_chunks[:5]]

        dataset_rows = [{
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts_str,
            "reference": ground_truth or "",
        }]

        ds = EvaluationDataset.from_list(dataset_rows)
        eval_result = evaluate(ds, metrics=metrics)

        # Extract scores from result.scores (list of dicts)
        if eval_result.scores and len(eval_result.scores) > 0:
            scores = eval_result.scores[0]

            def safe_float(val: Any) -> float:
                if val is None:
                    return float("nan")
                try:
                    f = float(val)
                    return f if not np.isnan(f) else float("nan")
                except (TypeError, ValueError):
                    return float("nan")

            result.faithfulness = safe_float(scores.get("faithfulness"))
            result.answer_relevancy = safe_float(scores.get("answer_relevancy"))
            result.__post_init__()

            # Composite score: trung bình 2 metrics (faithfulness + answer_relevancy)
            valid = [v for v in [result.faithfulness, result.answer_relevancy]
                     if not np.isnan(v)]
            result.ragas_score = float(np.mean(valid)) if valid else float("nan")

            logger.info(f"RAGAS eval done: faithfulness={result.faithfulness:.3f}, "
                        f"relevancy={result.answer_relevancy:.3f}, "
                        f"ragas_score={result.ragas_score:.3f}")
        else:
            result.eval_error = "No scores returned"

    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            result.eval_error = f"API quota exceeded: {err_str[:100]}"
            logger.warning(result.eval_error)
        elif "503" in err_str or "UNAVAILABLE" in err_str:
            result.eval_error = f"API temporarily unavailable: {err_str[:100]}"
            logger.warning(result.eval_error)
        else:
            result.eval_error = f"Eval error: {err_str[:150]}"
            logger.warning(result.eval_error)

    return result


def evaluate_dataset(
    questions: list[str],
    answers: list[str],
    retrieved_chunks_list: list[list[dict[str, Any]]],
    ground_truths: Optional[list[str]] = None,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """
    Đánh giá nhiều questions cùng lúc bằng RAGAS.
    Trả về dict với average scores + per-query results.
    """
    if not questions:
        return {"num_questions": 0}

    results: list[RAGEvaluationResult] = []
    for i, (q, a, chunks) in enumerate(zip(questions, answers, retrieved_chunks_list)):
        gt = ground_truths[i] if ground_truths else None
        res = evaluate_rag_response(q, a, chunks, gt, model=model)
        results.append(res)

    # Aggregate
    def nanmean(vals: list[float]) -> float:
        valid = [v for v in vals if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    agg = DatasetEvaluationResult(
        results=results,
        num_questions=len(results),
        avg_faithfulness=nanmean([r.faithfulness for r in results]),
        avg_answer_relevancy=nanmean([r.answer_relevancy for r in results]),
    )

    all_scores = [r.ragas_score for r in results if not np.isnan(r.ragas_score)]
    agg.avg_ragas_score = float(np.mean(all_scores)) if all_scores else float("nan")

    logger.info(f"RAGAS dataset eval done: {agg.num_questions} questions, "
                f"avg_ragas={agg.avg_ragas_score:.3f}, "
                f"avg_faithfulness={agg.avg_faithfulness:.3f}")

    # Legacy dict format để tương thích
    return {
        "avg_retrieval_precision": float("nan"),
        "avg_retrieval_recall": float("nan"),
        "avg_answer_relevance": agg.avg_answer_relevancy,
        "avg_answer_faithfulness": agg.avg_faithfulness,
        "avg_ragas_score": agg.avg_ragas_score,
        "num_questions": agg.num_questions,
        "per_query": [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers (dùng keyword-overlap khi RAGAS không khả dụng)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    relevant_chunks: Optional[list[dict[str, Any]]] = None,
) -> tuple[float, float]:
    """Fallback: keyword-overlap precision/recall khi RAGAS không khả dụng."""
    import re

    if not retrieved_chunks:
        return 0.0, 0.0

    query_keywords = set(re.findall(r"\b\w{3,}\b", query.lower()))
    retrieved_texts = [c.get("text", "") for c in retrieved_chunks]
    retrieved_keywords = set()
    for text in retrieved_texts:
        retrieved_keywords.update(re.findall(r"\b\w{3,}\b", text.lower()))

    overlap = query_keywords & retrieved_keywords
    precision = len(overlap) / len(query_keywords) if query_keywords else 0.0

    if relevant_chunks:
        relevant_text = " ".join(c.get("text", "") for c in relevant_chunks)
        relevant_keywords = set(re.findall(r"\b\w{3,}\b", relevant_text.lower()))
        recall = len(overlap & relevant_keywords) / len(relevant_keywords) if relevant_keywords else 0.0
    else:
        recall = precision

    return round(precision, 3), round(recall, 3)


def compute_answer_relevance(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> float:
    """Fallback: keyword-overlap relevance khi RAGAS không khả dụng."""
    import re

    if not answer or not retrieved_chunks:
        return 0.0

    answer_keywords = set(re.findall(r"\b\w{3,}\b", answer.lower()))
    context_keywords = set()
    for chunk in retrieved_chunks:
        context_keywords.update(re.findall(r"\b\w{3,}\b", chunk.get("text", "").lower()))

    overlap = answer_keywords & context_keywords
    relevance = len(overlap) / len(answer_keywords) if answer_keywords else 0.0
    return round(relevance, 3)


def compute_answer_faithfulness(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> float:
    """Fallback: entity-pattern faithfulness khi RAGAS không khả dụng."""
    import re

    if not answer:
        return 0.0

    context_text = " ".join(c.get("text", "") for c in retrieved_chunks)
    context_lower = context_text.lower()
    answer_lower = answer.lower()

    entity_patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{1,2} giờ\d{0,3}",
        r"Phòng [A-Z]\d{3}",
        r"Khu [A-Z]",
        r"Thứ [A-Za-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+",
    ]

    found = total = 0
    for pattern in entity_patterns:
        matches = re.findall(pattern, answer_lower)
        for match in matches:
            total += 1
            if match in context_lower:
                found += 1

    return 1.0 if total == 0 else round(found / total, 3)


def compute_ragas_score(
    precision: float,
    recall: float,
    relevance: float,
    faithfulness: float,
) -> float:
    """Fallback: simple composite score khi RAGAS không khả dụng."""
    score = 0.25 * precision + 0.25 * recall + 0.25 * relevance + 0.25 * faithfulness
    return round(score, 3)


# ---------------------------------------------------------------------------
# LLM-based evaluation (Gemini grading — kept as separate function)
# ---------------------------------------------------------------------------

_llm_evaluation_prompt = """Bạn là chuyên gia đánh giá chất lượng câu trả lời RAG.

Nhiệm vụ: Đánh giá câu trả lời dựa trên ngữ cảnh được truy xuất.

**Ngữ cảnh truy xuất:**
{context}

**Câu hỏi:**
{question}

**Câu trả lời cần đánh giá:**
{answer}

Đánh giá theo thang 0.0-1.0:

1. FAITHFULNESS (trung thực): Câu trả lời có đúng sự thật với ngữ cảnh không? (1.0 = hoàn toàn đúng, 0.0 = hoàn toàn sai/bịa đặt)
2. RELEVANCE (liên quan): Câu trả lời có trả lời đúng câu hỏi không? (1.0 = đầy đủ, 0.0 = không liên quan)

Trả về JSON:
{{"faithfulness": 0.0-1.0, "relevance": 0.0-1.0, "reason": "giải thích ngắn"}}
"""


@dataclass
class LLMEvaluationResult:
    """Kết quả đánh giá bằng LLM (Gemini direct call)."""
    faithfulness: float = 0.0
    relevance: float = 0.0
    reason: str = ""


def evaluate_with_llm(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str = "gemini-2.5-flash",
) -> LLMEvaluationResult:
    """
    Dùng Gemini trực tiếp để đánh giá faithfulness và relevance.
    Dùng khi RAGAS không khả dụng hoặc cần đánh giá nhanh.
    """
    if not retrieved_chunks:
        return LLMEvaluationResult(faithfulness=0.0, relevance=0.0, reason="No retrieved context")

    context_parts = []
    for r in retrieved_chunks[:5]:
        text = r.get("text", "")
        if text:
            context_parts.append(text[:300])
    context_str = "\n---\n".join(context_parts)

    prompt = _llm_evaluation_prompt.format(
        question=question,
        answer=answer,
        context=context_str,
    )

    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
        )
        text = response.text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        import json
        data = json.loads(text)
        return LLMEvaluationResult(
            faithfulness=float(data.get("faithfulness", 0.0)),
            relevance=float(data.get("relevance", 0.0)),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        logger.warning(f"LLM evaluation failed: {e}")
        return LLMEvaluationResult(faithfulness=0.0, relevance=0.0, reason=f"Error: {e}")
