r"""
run_benchmark.py -- Chạy benchmark 20 queries tren he thong RAG.
Output: JSON file voi per-query results + aggregate metrics.
Usage:
  .venv\Scripts\python.exe tests/run_benchmark.py --output tests/baseline_metrics.json
"""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.benchmark_queries import BENCHMARK_QUERIES
import math


def _nanmean(vals: list) -> float:
    """Mean ignoring NaN values."""
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(valid) / len(valid), 3) if valid else 0.0


# ---------------------------------------------------------------------------
# QueryParser: uses the LLM-based QueryParser from src/query_parser/.
# This is the NEW system - replaces the old regex-based modules.
# After Step 3 integration, this parser is also wired into RAGPipeline.
# ---------------------------------------------------------------------------

class _CurrentSystemQueryParser:
    """
    Wraps the LLM-based QueryParser into a .parse() interface.
    After Step 2, this is the QueryParser itself (not a shim).
    Used for all measurements - the baseline was captured before this
    QueryParser was implemented, so delta shows the improvement.
    """

    def __init__(self):
        from src.query_parser import QueryParser
        self._parser = QueryParser(model="gemini-2.5-flash", temperature=0.0)

    def parse(self, query: str):
        return self._parser.parse(query)


def _temporal_correct(parsed, expected):
    """So sanh temporal parsed vs expected. Tra ve tuple (bool, str_reason)."""
    if expected.get("expected_year"):
        ok = parsed.temporal.year == expected["expected_year"]
        return ok, f"year={parsed.temporal.year} vs expected={expected['expected_year']}"
    elif expected.get("expected_month") in ("current", "previous"):
        ok = parsed.temporal.has_reference
        return ok, f"has_reference={parsed.temporal.has_reference}"
    elif expected.get("expected_week"):
        ok = parsed.temporal.has_reference
        return ok, f"has_reference={parsed.temporal.has_reference}, type={parsed.temporal.type}"
    elif expected.get("expected_month"):
        ok = parsed.temporal.month == expected["expected_month"]
        return ok, f"month={parsed.temporal.month} vs expected={expected['expected_month']}"
    elif expected.get("expected_dow"):
        ok = parsed.temporal.day_of_week == expected["expected_dow"]
        return ok, f"dow={parsed.temporal.day_of_week} vs expected={expected['expected_dow']}"
    else:
        return True, "no temporal expected"


def _content_correct(parsed, expected):
    """So sanh content parsed vs expected. Tra ve tuple (bool, str_reason)."""
    if expected.get("expected_chairperson"):
        chairperson = (parsed.content.chairperson or "").lower()
        ok = expected["expected_chairperson"].lower() in chairperson
        return ok, f"chairperson='{parsed.content.chairperson}' vs expected='{expected['expected_chairperson']}'"
    elif expected.get("expected_event"):
        event_name = (parsed.content.event_name or "").lower()
        ok = expected["expected_event"].lower() in event_name
        return ok, f"event_name='{parsed.content.event_name}' vs expected='{expected['expected_event']}'"
    else:
        return True, "no content filter expected"


# 5 representative queries cho --quick mode (tối ưu quota: 5 queries × 2 metrics = ~10 LLM calls)
QUICK_QUERIES = [1, 2, 3, 4, 7]  # IDs 1,2,3,4,7 từ benchmark_queries.json


def run_benchmark(output_path: str, quick: bool = False, run_eval: bool = True):
    from src.main import RAGPipeline
    from src.evaluation import evaluate_rag_response

    pipe = RAGPipeline()
    pipe._ensure_loaded()
    results = []

    queries = [q for q in BENCHMARK_QUERIES if q["id"] in QUICK_QUERIES] if quick else BENCHMARK_QUERIES

    for item in queries:
        qid = item["id"]
        query = item["query"]
        expected = item

        # --- Query Parser ---
        t0 = time.perf_counter()
        parsed = pipe._query_parser.parse(query)
        parse_ms = (time.perf_counter() - t0) * 1000

        temporal_ok, temporal_reason = _temporal_correct(parsed, expected)
        content_ok, content_reason = _content_correct(parsed, expected)
        query_type_ok = parsed.query_type.value == expected.get("expected_type", "list")

        # --- Retrieval ---
        t1 = time.perf_counter()
        try:
            hits, scope, meta = pipe.hybrid_search(query, top_k=20)
            retrieval_ms = (time.perf_counter() - t1) * 1000
            hit_rate_5 = 1 if len(hits) >= 5 else 0
            hit_rate_10 = 1 if len(hits) >= 10 else 0
            top_score = hits[0]["score"] if hits else 0.0
            num_results = len(hits)
            no_answer = False
            retrieval_error = None
        except Exception as e:
            retrieval_ms = (time.perf_counter() - t1) * 1000
            hit_rate_5 = hit_rate_10 = 0
            top_score = 0.0
            num_results = 0
            no_answer = True
            hits = []
            retrieval_error = str(e)[:120]

        # --- Generation ---
        t2 = time.perf_counter()
        answer = ""
        gen_error = None
        try:
            # Retry logic cho 429 quota errors
            last_err = None
            for attempt in range(3):
                try:
                    answer, _, _, _ = pipe.ask(query)
                    last_err = None
                    break
                except Exception as e:
                    err_str = str(e)
                    last_err = err_str
                    if "429" in err_str and attempt < 2:
                        time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
                        continue
                    raise
            if last_err and not answer:
                answer = f"ERROR: {last_err}"
                gen_error = str(last_err)[:120]
            gen_ms = (time.perf_counter() - t2) * 1000
        except Exception as e:
            gen_ms = (time.perf_counter() - t2) * 1000
            if not answer:
                answer = f"ERROR: {e}"
            gen_error = str(e)[:120]

        # --- RAGAS Evaluation (2 metrics: faithfulness + answer_relevancy) ---
        eval_ms = 0.0
        faithfulness = float("nan")
        answer_relevancy = float("nan")
        ragas_score = float("nan")
        eval_error = None

        if run_eval and hits and answer and not retrieval_error and not gen_error:
            import time as _time_module
            t_eval = _time_module.perf_counter()
            try:
                eval_result = evaluate_rag_response(
                    question=query,
                    answer=answer,
                    retrieved_chunks=hits,
                    ground_truth=None,
                    model="gemini-2.5-flash",
                )
                eval_ms = (_time_module.perf_counter() - t_eval) * 1000

                faithfulness = eval_result.faithfulness
                answer_relevancy = eval_result.answer_relevancy
                ragas_score = eval_result.ragas_score
                eval_error = eval_result.eval_error
            except Exception as e:
                eval_ms = 0.0
                eval_error = str(e)[:120]

        results.append({
            "id": qid,
            "query": query,
            "type": expected["type"],
            # Parser
            "parse_ms": round(parse_ms, 1),
            "temporal_correct": temporal_ok,
            "temporal_reason": temporal_reason,
            "content_correct": content_ok,
            "content_reason": content_reason,
            "query_type_correct": query_type_ok,
            "query_type_parsed": parsed.query_type.value,
            "confidence": round(parsed.confidence, 3),
            "is_general_query": parsed.is_general_query,
            # Retrieval
            "retrieval_ms": round(retrieval_ms, 1),
            "retrieval_error": retrieval_error,
            "hit_rate_5": hit_rate_5,
            "hit_rate_10": hit_rate_10,
            "top_score": round(top_score, 4),
            "num_results": num_results,
            "no_answer": no_answer,
            # Generation
            "gen_ms": round(gen_ms, 1),
            "gen_error": gen_error,
            "answer_length": len(answer),
            "answer_preview": answer[:150] if answer else "",
            # RAGAS Evaluation (faithfulness + answer_relevancy only)
            "eval_ms": round(eval_ms, 1),
            "faithfulness": float(faithfulness),
            "answer_relevancy": float(answer_relevancy),
            "ragas_score": float(ragas_score),
            "eval_error": eval_error,
            # E2E
            "e2e_ms": round(parse_ms + retrieval_ms + gen_ms + eval_ms, 1),
        })

    # --- Aggregate ---
    n = len(results)
    metrics = {
        # Parser
        "Parser - Temporal Detection Rate": round(sum(r["temporal_correct"] for r in results) / n, 4),
        "Parser - Content Detection Rate": round(sum(r["content_correct"] for r in results) / n, 4),
        "Parser - Query Type Accuracy": round(sum(r["query_type_correct"] for r in results) / n, 4),
        "Parser - Avg Confidence": round(sum(r["confidence"] for r in results) / n, 3),
        "Parser - Avg Latency (ms)": round(sum(r["parse_ms"] for r in results) / n, 1),
        "Parser - Max Latency (ms)": round(max(r["parse_ms"] for r in results), 1),
        # Retrieval
        "Retrieval - Hit-Rate@5": round(sum(r["hit_rate_5"] for r in results) / n, 4),
        "Retrieval - Hit-Rate@10": round(sum(r["hit_rate_10"] for r in results) / n, 4),
        "Retrieval - Avg Latency (ms)": round(sum(r["retrieval_ms"] for r in results) / n, 1),
        "Retrieval - Max Latency (ms)": round(max(r["retrieval_ms"] for r in results), 1),
        "Retrieval - No-Answer Rate": round(sum(r["no_answer"] for r in results) / n, 4),
        "Retrieval - Avg Results": round(sum(r["num_results"] for r in results) / n, 1),
        "Retrieval - Avg Top Score": round(sum(r["top_score"] for r in results) / n, 4),
        # Generation
        "Generation - Avg Latency (ms)": round(sum(r["gen_ms"] for r in results) / n, 1),
        "Generation - Max Latency (ms)": round(max(r["gen_ms"] for r in results), 1),
        "Generation - Avg Answer Length": round(sum(r["answer_length"] for r in results) / n, 1),
        # RAGAS Evaluation (faithfulness + answer_relevancy)
        "RAGAS - Avg Faithfulness": _nanmean([r["faithfulness"] for r in results]),
        "RAGAS - Avg Answer Relevancy": _nanmean([r["answer_relevancy"] for r in results]),
        "RAGAS - Avg Score": _nanmean([r["ragas_score"] for r in results]),
        "RAGAS - Eval Errors": sum(1 for r in results if r["eval_error"]),
        "RAGAS - Avg Eval Latency (ms)": round(sum(r["eval_ms"] for r in results) / n, 1),
        # E2E
        "E2E - Avg Latency (ms)": round(sum(r["e2e_ms"] for r in results) / n, 1),
        "E2E - P95 Latency (ms)": round(sorted(r["e2e_ms"] for r in results)[int(n * 0.95)], 1),
        # Counts
        "total_queries": n,
        "retrieval_errors": sum(1 for r in results if r["retrieval_error"]),
        "generation_errors": sum(1 for r in results if r["gen_error"]),
        "per_query": results,
    }
    json.dump(
        metrics,
        open(output_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/benchmark_results.json")
    parser.add_argument("--quick", action="store_true",
                        help="Chạy 5 representative queries thay vì 20 (tiết kiệm quota API)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Bỏ qua RAGAS evaluation (tiết kiệm quota API)")
    args = parser.parse_args()

    queries_to_run = [q for q in BENCHMARK_QUERIES if q["id"] in QUICK_QUERIES] if args.quick else BENCHMARK_QUERIES
    print(f"Running benchmark ({len(queries_to_run)} queries)...")
    metrics = run_benchmark(args.output, quick=args.quick, run_eval=not args.no_eval)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS -> {args.output}")
    print(f"{'='*60}")
    print(f"  Parser Temporal Rate:  {metrics['Parser - Temporal Detection Rate']:.1%}")
    print(f"  Parser Content Rate:   {metrics['Parser - Content Detection Rate']:.1%}")
    print(f"  Parser QueryType Rate: {metrics['Parser - Query Type Accuracy']:.1%}")
    print(f"  Parser Avg Latency:    {metrics['Parser - Avg Latency (ms)']:.0f}ms")
    print(f"  Retrieval Hit@5:       {metrics['Retrieval - Hit-Rate@5']:.1%}")
    print(f"  Retrieval Hit@10:      {metrics['Retrieval - Hit-Rate@10']:.1%}")
    print(f"  Retrieval Avg Latency: {metrics['Retrieval - Avg Latency (ms)']:.0f}ms")
    print(f"  RAGAS Faithfulness:   {metrics['RAGAS - Avg Faithfulness']:.3f}")
    print(f"  RAGAS Relevancy:     {metrics['RAGAS - Avg Answer Relevancy']:.3f}")
    print(f"  RAGAS Score:         {metrics['RAGAS - Avg Score']:.3f}")
    print(f"  RAGAS Eval Errors:    {metrics['RAGAS - Eval Errors']}")
    print(f"  E2E Avg Latency:       {metrics['E2E - Avg Latency (ms)']:.0f}ms")
    print(f"  Retrieval Errors:      {metrics['retrieval_errors']}")
    print(f"  Generation Errors:    {metrics['generation_errors']}")
    print(f"{'='*60}")
