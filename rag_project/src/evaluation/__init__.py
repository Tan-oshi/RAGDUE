"""
Evaluation module - RAG quality assessment using RAGAS 0.4.x.
Primary: RAGAS LLM-based metrics (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall).
Fallback: keyword-overlap metrics when RAGAS LLM is unavailable.
"""
from .metrics import (
    # Result dataclasses
    RAGEvaluationResult,
    DatasetEvaluationResult,
    LLMEvaluationResult,
    # Primary evaluation (RAGAS)
    evaluate_rag_response,
    evaluate_dataset,
    # Legacy / fallback (keyword-overlap)
    compute_retrieval_metrics,
    compute_answer_relevance,
    compute_answer_faithfulness,
    compute_ragas_score,
    # LLM direct grading
    evaluate_with_llm,
    # RAGAS client management
    reset_ragas_client,
    _get_ragas_llm,
    _get_ragas_embeddings,
)

__all__ = [
    # Dataclasses
    "RAGEvaluationResult",
    "DatasetEvaluationResult",
    "LLMEvaluationResult",
    # Primary (RAGAS)
    "evaluate_rag_response",
    "evaluate_dataset",
    # Legacy
    "compute_retrieval_metrics",
    "compute_answer_relevance",
    "compute_answer_faithfulness",
    "compute_ragas_score",
    # Direct LLM
    "evaluate_with_llm",
    # Utilities
    "reset_ragas_client",
    "_get_ragas_llm",
    "_get_ragas_embeddings",
]
