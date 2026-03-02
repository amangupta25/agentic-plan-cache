#!/usr/bin/env python3
"""Benchmark: APC (cache-enabled) vs Baseline (no cache).

Runs both agents on the same query set and prints a side-by-side comparison
to quantify cost and latency savings from plan caching.

Usage:
    source .env && .venv/bin/python demo/benchmark.py
"""

from __future__ import annotations

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apc.agent import APCAgent
from apc.baseline import BaselineAgent
from apc.cache.plan_cache import PlanCache
from apc.llm.openai_provider import OpenAIProvider
from apc.tools.calculator import CalculatorTool
from apc.tools.registry import ToolRegistry
from apc.tools.web_search import WebSearchTool

# ---------------------------------------------------------------------------
# Queries — mix of similar (to exercise caching) and diverse
# ---------------------------------------------------------------------------
QUERIES = [
    # Group 1: similar math queries (sqrt + addition)
    "What is the square root of 144 plus 12?",
    "What is the square root of 256 plus 20?",
    "What is the square root of 81 plus 5?",
    # Group 2: similar search queries (physics constants)
    "What is the speed of light in m/s?",
    "What is the gravitational constant G?",
    # Group 3: unique / different pattern
    "Calculate 2 raised to the power of 10 then subtract 24.",
]


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    return registry


def run_benchmark() -> None:
    expensive_model = os.getenv("APC_EXPENSIVE_MODEL", "gpt-4o")
    cheap_model = os.getenv("APC_CHEAP_MODEL", "gpt-4o-mini")

    print("=" * 72)
    print("APC vs Baseline Benchmark")
    print("=" * 72)
    print(f"  Expensive LLM : {expensive_model}")
    print(f"  Cheap LLM     : {cheap_model}")
    print(f"  Queries       : {len(QUERIES)}")
    print()

    expensive_llm = OpenAIProvider(model=expensive_model)
    cheap_llm = OpenAIProvider(model=cheap_model)

    # ----- Baseline (no cache, expensive LLM only) -----
    print("Running BASELINE (no cache) ...")
    baseline = BaselineAgent(
        llm=expensive_llm,
        tool_registry=build_tool_registry(),
    )
    baseline_results = []
    for q in QUERIES:
        baseline_results.append(baseline.run(q))

    # ----- APC (cache-enabled, fresh cache each run) -----
    print("Running APC     (cache-enabled) ...")
    apc = APCAgent(
        expensive_llm=expensive_llm,
        cheap_llm=cheap_llm,
        tool_registry=build_tool_registry(),
        cache=PlanCache(max_size=128),  # fresh in-memory cache
    )
    apc_results = []
    for q in QUERIES:
        apc_results.append(apc.run(q))

    # ----- Per-query results table -----
    print()
    print("=" * 72)
    print("Per-Query Results")
    print("=" * 72)

    hdr = (
        f"{'#':<3} {'Query':<45} {'BL ms':>7} {'APC ms':>7} "
        f"{'Speed':>6} {'BL LLM':>6} {'APC LLM':>7} {'Cache':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for i, (br, ar) in enumerate(zip(baseline_results, apc_results), 1):
        speedup = br.latency_ms / ar.latency_ms if ar.latency_ms else float("inf")
        cache_str = "HIT" if ar.cache_hit else "MISS"
        query_short = br.query if len(br.query) <= 44 else br.query[:41] + "..."
        print(
            f"{i:<3} {query_short:<45} {br.latency_ms:>7.0f} {ar.latency_ms:>7.0f} "
            f"{speedup:>5.1f}x {br.llm_calls:>6} {ar.llm_calls:>7} {cache_str:>6}"
        )
        if ar.keyword:
            print(f"    Keyword: {ar.keyword}")

    # ----- Summary -----
    total_bl_ms = sum(r.latency_ms for r in baseline_results)
    total_apc_ms = sum(r.latency_ms for r in apc_results)
    total_bl_llm = sum(r.llm_calls for r in baseline_results)
    total_apc_llm = sum(r.llm_calls for r in apc_results)

    # Expensive LLM calls:
    #   Baseline: 2 per query (plan + synthesize), all expensive
    #   APC: 1 per cache miss (plan only), rest are cheap
    bl_expensive = total_bl_llm  # all calls use expensive LLM
    apc_expensive = apc.stats.cache_misses  # 1 expensive call per miss (planning)

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total latency (baseline) : {total_bl_ms:>10.0f} ms")
    print(f"  Total latency (APC)      : {total_apc_ms:>10.0f} ms")
    if total_bl_ms:
        pct_lat = (1 - total_apc_ms / total_bl_ms) * 100
        print(f"  Latency reduction        : {pct_lat:>9.0f}%")

    print()
    print(f"  Total LLM calls (baseline) : {total_bl_llm:>6}")
    print(f"  Total LLM calls (APC)      : {total_apc_llm:>6}")

    print()
    print(f"  Expensive LLM calls (baseline) : {bl_expensive:>4}")
    print(f"  Expensive LLM calls (APC)      : {apc_expensive:>4}")
    if bl_expensive:
        pct_exp = (1 - apc_expensive / bl_expensive) * 100
        print(f"  Expensive call reduction       : {pct_exp:.0f}%")

    print()
    print(f"  APC cache hits   : {apc.stats.cache_hits}")
    print(f"  APC cache misses : {apc.stats.cache_misses}")
    print(f"  APC hit rate     : {apc.stats.hit_rate:.0%}")

    # ----- Answer comparison -----
    print()
    print("=" * 72)
    print("Answer Comparison")
    print("=" * 72)
    for i, (br, ar) in enumerate(zip(baseline_results, apc_results), 1):
        cache_str = "HIT" if ar.cache_hit else "MISS"
        print(f"\n  Q{i} [{cache_str}]: {br.query}")
        print(f"    Baseline : {br.final_answer}")
        print(f"    APC      : {ar.final_answer}")
        match = br.final_answer.strip() == ar.final_answer.strip()
        print(f"    Match    : {'YES' if match else 'NO'}")
    print()


if __name__ == "__main__":
    run_benchmark()
