"""
analyze_results.py - Phan tich va truc quan hoa ket qua benchmark.
So sanh baseline vs post-refactor, hien thi bang per-query, bottleneck analysis.
Usage:
  .venv\Scripts\python.exe tests/analyze_results.py --baseline tests/baseline_metrics.json --current tests/post_refactor_metrics.json
  .venv\Scripts\python.exe tests/analyze_results.py --current tests/baseline_metrics.json
"""
import json
import argparse
from pathlib import Path


def _g(d: dict, key: str, default):
    """Try both new format (' - ') and old format (' — ') keys."""
    v = d.get(key, None)
    if v is not None:
        return v
    # Try old em-dash format
    old_key = key.replace(" - ", " — ")
    return d.get(old_key, default)


def ascii_bar(value: float, width: int = 30) -> str:
    """Vẽ ASCII bar cho giá trị 0.0–1.0."""
    filled = int(value * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {value:.1%}"


GATE_DEFAULTS = {
    "parser_temporal_gate": 0.90,
    "parser_content_gate": 0.80,
    "parser_qtype_gate": 0.85,
    "parser_latency_gate": 500,
    "retrieval_hit5_gate": 0.80,
    "retrieval_hit10_gate": 0.90,
    "retrieval_latency_gate": 400,
    "retrieval_noanswer_gate": 0.10,
    "e2e_latency_gate": 3000,
    "ragas_faithfulness_gate": 0.80,
    "ragas_relevancy_gate": 0.70,
    "ragas_score_gate": 0.70,
}


def print_comparison_table(baseline: dict, current: dict, title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Metric':<38} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>8}")
    print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    metrics_to_compare = [
        ("Parser - Temporal Detection Rate", "rate"),
        ("Parser - Content Detection Rate", "rate"),
        ("Parser - Query Type Accuracy", "rate"),
        ("Parser - Avg Confidence", "number"),
        ("Parser - Avg Latency (ms)", "latency"),
        ("Parser - Max Latency (ms)", "latency"),
        ("Retrieval - Hit-Rate@5", "rate"),
        ("Retrieval - Hit-Rate@10", "rate"),
        ("Retrieval - Avg Latency (ms)", "latency"),
        ("Retrieval - Max Latency (ms)", "latency"),
        ("Retrieval - No-Answer Rate", "rate"),
        ("Retrieval - Avg Results", "number"),
        ("Retrieval - Avg Top Score", "number"),
        ("Generation - Avg Latency (ms)", "latency"),
        ("Generation - Max Latency (ms)", "latency"),
        # RAGAS Evaluation (faithfulness + answer_relevancy)
        ("RAGAS - Avg Faithfulness", "rate"),
        ("RAGAS - Avg Answer Relevancy", "rate"),
        ("RAGAS - Avg Score", "rate"),
        ("RAGAS - Avg Eval Latency (ms)", "latency"),
        ("E2E - Avg Latency (ms)", "latency"),
        ("E2E - P95 Latency (ms)", "latency"),
    ]

    for metric_name, mtype in metrics_to_compare:
        b = _g(baseline, metric_name, 0)
        c = _g(current, metric_name, 0)
        if mtype == "rate":
            delta = (c - b) * 100
            delta_str = f"{delta:+.1f}pp"
            if "Rate" in metric_name or "Accuracy" in metric_name:
                status = "PASS" if c >= b else "FAIL"
            else:
                status = "PASS" if c <= b else "WARN"
        else:
            delta = c - b
            delta_str = f"{delta:+.0f}ms"
            status = "PASS" if (("Latency" in metric_name and c <= b * 1.2) or c <= b) else "FAIL"

        bar = ascii_bar(c) if mtype == "rate" else ""
        print(f"  {metric_name:<38} {b:>10} {c:>10} {delta_str:>10} {status:>8}  {bar}")

    print(f"{'='*80}")


def print_per_query_table(current: dict, baseline: dict = None, failed_only: bool = False):
    print(f"\n{'-'*100}")
    print(f"  PER-QUERY BREAKDOWN")
    print(f"{'-'*100}")
    print(f"  {'ID':>3} {'Temporal':>9} {'Content':>9} {'QType':>9} {'Hit@5':>6} {'Hit@10':>7} {'Faith':>5} {'Rel':>5} {'E2E(ms)':>8} {'Err':>5}")
    print(f"  {'-'*3} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*8} {'-'*5}")

    for r in current["per_query"]:
        if failed_only:
            if (
                r["temporal_correct"]
                and r["content_correct"]
                and r["hit_rate_5"]
                and not r["retrieval_error"]
                and not r["gen_error"]
            ):
                continue

        temporal = "PASS" if r["temporal_correct"] else "FAIL"
        content = "PASS" if r["content_correct"] else "FAIL"
        qtype = "PASS" if r["query_type_correct"] else "FAIL"
        hit5 = "PASS" if r["hit_rate_5"] else "FAIL"
        hit10 = "PASS" if r["hit_rate_10"] else "FAIL"
        err = "R" if r["retrieval_error"] else ("G" if r["gen_error"] else ("E" if r.get("eval_error") else ""))

        if baseline:
            b = next((x for x in baseline["per_query"] if x["id"] == r["id"]), None)
            if b:
                b_hit5 = "PASS" if b["hit_rate_5"] else "FAIL"
                hit5 = f"{b_hit5}->{hit5}"
                hit10 = f"{'PASS' if b['hit_rate_10'] else 'FAIL'}->{hit10}"

        # Format RAGAS scores
        import math
        def fmt(v):
            if v is None: return "N/A"
            if isinstance(v, float) and math.isnan(v): return "NaN"
            return f"{v:.2f}"
        faith = fmt(r.get("faithfulness"))
        rel = fmt(r.get("answer_relevancy"))

        # Truncate query to ASCII-safe subset
        query_short = r["query"][:35]
        query_short = "".join(c if ord(c) < 128 else "?" for c in query_short)
        print(f"  {r['id']:>3} {temporal:>9} {content:>9} {qtype:>9} {hit5:>6} {hit10:>7} {faith:>5} {rel:>5} {r['e2e_ms']:>8.0f} {err:>5}  {query_short}")


def print_error_summary(current: dict):
    errors = []
    for r in current["per_query"]:
        q = "".join(c if ord(c) < 128 else "?" for c in r["query"])
        if r["retrieval_error"]:
            errors.append(f"  Q{r['id']} [RETRIEVAL]: {r['retrieval_error']} - '{q}'")
        if r["gen_error"]:
            errors.append(f"  Q{r['id']} [GENERATION]: {r['gen_error']} - '{q}'")
        if r.get("eval_error"):
            errors.append(f"  Q{r['id']} [RAGAS]: {r['eval_error']} - '{q}'")

    if errors:
        print(f"\n{'!'*80}")
        print(f"  ERRORS ({len(errors)})")
        print(f"{'!'*80}")
        for e in errors:
            print(e)
    else:
        print(f"\n  [OK] No errors detected")


def print_gate_summary(current: dict, gates: dict):
    """Kiểm tra từng metric có pass gate khong."""
    print(f"\n{'+'*60}")
    print(f"  GATE VERIFICATION")
    print(f"  {'+'*60}")
    print(f"  {'Metric':<45} {'Actual':>8} {'Gate':>8} {'Result':>8}")
    print(f"  {'+'*45} {'+'*8} {'+'*8} {'+'*8}")

    gate_checks = [
        (
            "Parser - Temporal Detection Rate",
            current.get("Parser - Temporal Detection Rate", 0),
            gates.get("parser_temporal_gate", 0.90),
        ),
        (
            "Parser - Content Detection Rate",
            current.get("Parser - Content Detection Rate", 0),
            gates.get("parser_content_gate", 0.80),
        ),
        (
            "Parser - Query Type Accuracy",
            current.get("Parser - Query Type Accuracy", 0),
            gates.get("parser_qtype_gate", 0.85),
        ),
        (
            "Parser - Avg Latency (ms)",
            current.get("Parser - Avg Latency (ms)", 9999),
            gates.get("parser_latency_gate", 500),
        ),
        (
            "Retrieval - Hit-Rate@5",
            current.get("Retrieval - Hit-Rate@5", 0),
            gates.get("retrieval_hit5_gate", 0.80),
        ),
        (
            "Retrieval - Hit-Rate@10",
            current.get("Retrieval - Hit-Rate@10", 0),
            gates.get("retrieval_hit10_gate", 0.90),
        ),
        (
            "Retrieval - Avg Latency (ms)",
            current.get("Retrieval - Avg Latency (ms)", 9999),
            gates.get("retrieval_latency_gate", 400),
        ),
        (
            "Retrieval - No-Answer Rate",
            current.get("Retrieval - No-Answer Rate", 1),
            gates.get("retrieval_noanswer_gate", 0.10),
        ),
        (
            "E2E - Avg Latency (ms)",
            current.get("E2E - Avg Latency (ms)", 9999),
            gates.get("e2e_latency_gate", 3000),
        ),
        # RAGAS gates
        (
            "RAGAS - Avg Faithfulness",
            current.get("RAGAS - Avg Faithfulness", 0),
            gates.get("ragas_faithfulness_gate", 0.80),
        ),
        (
            "RAGAS - Avg Answer Relevancy",
            current.get("RAGAS - Avg Answer Relevancy", 0),
            gates.get("ragas_relevancy_gate", 0.70),
        ),
        (
            "RAGAS - Avg Score",
            current.get("RAGAS - Avg Score", 0),
            gates.get("ragas_score_gate", 0.70),
        ),
    ]

    all_passed = True
    for name, actual, gate in gate_checks:
        is_latency = "Latency" in name
        if is_latency:
            ok = actual <= gate
        else:
            ok = actual >= gate
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        if is_latency:
            print(f"  {name:<45} {actual:>7.0f}ms {gate:>7.0f}ms {status:>8}")
        else:
            print(f"  {name:<45} {actual:>7.1%} {gate:>7.1%} {status:>8}")

    print(f"  {'+'*45} {'+'*8} {'+'*8} {'+'*8}")

    # Error checks
    if current.get("retrieval_errors", 0) > 0:
        print(f"  FAIL: {current['retrieval_errors']} retrieval errors - MUST FIX before proceeding")
        all_passed = False
    if current.get("generation_errors", 0) > 0:
        print(f"  FAIL: {current['generation_errors']} generation errors - MUST FIX before proceeding")
        all_passed = False
    if current.get("RAGAS - Eval Errors", 0) > 0:
        print(f"  WARN: {current['RAGAS - Eval Errors']} RAGAS eval errors (may be quota issues)")

    print(f"\n  Overall: {'[PASS] ALL GATES PASSED' if all_passed else '[FAIL] SOME GATES FAILED - see above'}")
    return all_passed


def print_bottleneck_analysis(baseline: dict, current: dict, gates: dict):
    print(f"\n{'-'*80}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'-'*80}")

    table_data = [
        (
            "Parser Temporal Rate",
            gates.get("parser_temporal_gate", 0.90),
            "Parser - Temporal Detection Rate",
            "rate",
        ),
        (
            "Parser Content Rate",
            gates.get("parser_content_gate", 0.80),
            "Parser - Content Detection Rate",
            "rate",
        ),
        (
            "Hit-Rate@5",
            gates.get("retrieval_hit5_gate", 0.80),
            "Retrieval - Hit-Rate@5",
            "rate",
        ),
        (
            "Hit-Rate@10",
            gates.get("retrieval_hit10_gate", 0.90),
            "Retrieval - Hit-Rate@10",
            "rate",
        ),
        (
            "Retrieval Latency",
            gates.get("retrieval_latency_gate", 400),
            "Retrieval - Avg Latency (ms)",
            "latency",
        ),
        (
            "E2E Latency",
            gates.get("e2e_latency_gate", 3000),
            "E2E - Avg Latency (ms)",
            "latency",
        ),
        (
            "No-Answer Rate",
            gates.get("retrieval_noanswer_gate", 0.10),
            "Retrieval - No-Answer Rate",
            "rate",
        ),
    ]

    print(
        f"  {'Metric':<22} {'Baseline':>10} {'Current':>10} {'Delta':>8} {'ToTarget':>10} {'Diagnosis'}"
    )
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*30}")

    for metric, target, key, mtype in table_data:
        b = _g(baseline, key, 0)
        c = _g(current, key, 0)

        if mtype == "rate":
            delta = (c - b) * 100
            dist = (c - target) * 100
            delta_str = f"{delta:+.1f}pp"
            dist_str = f"{dist:+.1f}pp"
        else:
            delta = c - b
            dist = target - c
            delta_str = f"{delta:+.0f}ms"
            dist_str = f"{dist:+.0f}ms"

        flag = "[!]" if dist < 0 else "   "

        diagnosis = ""
        if "Temporal" in metric and c < target:
            diagnosis = "Parser temporal gaps"
        elif "Content" in metric and c < target:
            diagnosis = "Parser chairperson/event insufficient"
        elif "Hit" in metric and c < target:
            diagnosis = "Retrieval: top_k / RRF / filter logic"
        elif "Latency" in metric and c > target:
            diagnosis = "LLM overhead / network / parallel"
        elif "No-Answer" in metric and c > target:
            diagnosis = "Filter strict / data coverage"

        print(
            f"  {flag}{metric:<19} {b:>10} {c:>10} {delta_str:>8} {dist_str:>10}  {diagnosis}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="tests/baseline_metrics.json")
    parser.add_argument("--current", default="tests/current_metrics.json")
    parser.add_argument("--failed-only", action="store_true", help="Chi hien thi queries that bai")
    parser.add_argument("--gates", default="tests/gates.json")
    args = parser.parse_args()

    # Load gates
    gates_path = Path(args.gates)
    if gates_path.exists():
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
    else:
        gates = {}
    for k, v in GATE_DEFAULTS.items():
        gates.setdefault(k, v)

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    has_baseline = baseline_path.exists()
    has_current = current_path.exists()

    print(f"\n{'#'*80}")
    print(f"  BENCHMARK ANALYSIS")
    print(f"  Baseline: {args.baseline} ({'EXISTS' if has_baseline else 'NOT FOUND'})")
    print(f"  Current:  {args.current} ({'EXISTS' if has_current else 'NOT FOUND'})")
    print(f"{'#'*80}")

    baseline = None
    if has_baseline:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    if not has_current:
        print(f"\nERROR: Current results file not found: {args.current}")
        return

    current = json.loads(current_path.read_text(encoding="utf-8"))

    # Gate summary
    passed = print_gate_summary(current, gates)

    # Comparison
    if baseline:
        print_comparison_table(baseline, current, "BASELINE vs CURRENT")
        print_bottleneck_analysis(baseline, current, gates)

    # Per-query breakdown
    print_per_query_table(
        current, baseline if has_baseline else None, failed_only=args.failed_only
    )

    # Errors
    print_error_summary(current)

    print(f"\n{'='*80}")
    if passed:
        print(f"  [PASS] ALL GATES PASSED - proceed to next step")
    else:
        print(f"  [FAIL] GATES FAILED - fix issues before proceeding")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
