"""Benchmark queries — dùng chung cho run_benchmark.py và analyze_results.py."""
import json
from pathlib import Path

BENCHMARK_QUERIES = json.loads(
    Path(__file__).parent.joinpath("benchmark_queries.json").read_text(encoding="utf-8")
)
