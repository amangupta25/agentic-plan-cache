#!/usr/bin/env python3
"""Benchmark: multi-step workflows (3-5 steps each) — APC vs Baseline.

Tests the interleaved adapt→execute loop on workflows that chain
web_search → calculator → calculator, where later steps depend on
earlier results.

Usage:
    source .env && .venv/bin/python demo/multistep_benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apc.agent import APCAgent
from apc.baseline import BaselineAgent
from apc.cache.plan_cache import PlanCache
from apc.llm.openai_provider import OpenAIProvider
from apc.tools.calculator import CalculatorTool
from apc.tools.registry import ToolRegistry
from apc.tools.web_search import WebSearchTool

# ---------------------------------------------------------------------------
# Multi-step queries grouped by pattern
# ---------------------------------------------------------------------------
QUERIES = [
    # --- Group 1: search → compute → compute (distance = speed × time) ---
    "Find the speed of light in m/s, then calculate how far light travels in 5 seconds.",
    "Find the speed of light in m/s, then calculate how far light travels in 10 seconds.",
    "Find the speed of light in m/s, then calculate how far light travels in 3 seconds.",

    # --- Group 2: search → search → compute (combine two constants) ---
    "Look up the Planck constant and Avogadro's number, then multiply them together.",
    "Look up the gravitational constant and the Planck constant, then divide Planck by G.",

    # --- Group 3: search → compute → compute → compute (chain math on a constant) ---
    "Find the value of pi, compute pi squared, then take the square root of that result.",
    "Find Euler's number e, compute e squared, then subtract 1 from the result.",
    "Find the value of pi, compute pi cubed, then take the cube root of that result.",

    # --- Group 4: search → compute (unit conversion pattern) ---
    "Find the distance from Earth to the Sun in km, then convert it to miles by dividing by 1.609.",
    "Find the boiling point of water in Celsius, then convert it to Fahrenheit using the formula C * 9/5 + 32.",
    "Find the distance from Earth to the Sun in km, then convert it to meters by multiplying by 1000.",
]


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    return registry


def run_benchmark() -> None:
    expensive_model = os.getenv("APC_EXPENSIVE_MODEL", "gpt-4o")
    cheap_model = os.getenv("APC_CHEAP_MODEL", "gpt-4o-mini")

    print("=" * 80)
    print("Multi-Step Workflow Benchmark — APC vs Baseline")
    print("=" * 80)
    print(f"  Expensive LLM : {expensive_model}")
    print(f"  Cheap LLM     : {cheap_model}")
    print(f"  Queries       : {len(QUERIES)}")
    print(f"  Step range    : 2-5 steps per query (search + chained compute)")
    print()

    expensive_llm = OpenAIProvider(model=expensive_model)
    cheap_llm = OpenAIProvider(model=cheap_model)

    # ----- Baseline -----
    print("Running BASELINE (no cache) ...")
    baseline = BaselineAgent(
        llm=expensive_llm,
        tool_registry=build_tool_registry(),
    )
    baseline_results = []
    for q in QUERIES:
        baseline_results.append(baseline.run(q))

    # ----- APC -----
    print("Running APC     (cache-enabled, interleaved execution) ...")
    apc = APCAgent(
        expensive_llm=expensive_llm,
        cheap_llm=cheap_llm,
        tool_registry=build_tool_registry(),
        cache=PlanCache(max_size=128),
    )
    apc_results = []
    for q in QUERIES:
        apc_results.append(apc.run(q))

    # ----- Per-query table -----
    print()
    print("=" * 80)
    print("Per-Query Results")
    print("=" * 80)

    hdr = (
        f"{'#':<3} {'Query':<50} {'BL ms':>7} {'APC ms':>7} "
        f"{'Speed':>6} {'BL LLM':>6} {'APC LLM':>7} {'Steps':>5} {'Cache':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for i, (br, ar) in enumerate(zip(baseline_results, apc_results), 1):
        speedup = br.latency_ms / ar.latency_ms if ar.latency_ms else float("inf")
        cache_str = "HIT" if ar.cache_hit else "MISS"
        n_steps = len(ar.plan.steps)
        q_short = br.query if len(br.query) <= 49 else br.query[:46] + "..."
        print(
            f"{i:<3} {q_short:<50} {br.latency_ms:>7.0f} {ar.latency_ms:>7.0f} "
            f"{speedup:>5.1f}x {br.llm_calls:>6} {ar.llm_calls:>7} {n_steps:>5} {cache_str:>6}"
        )
        if ar.keyword:
            print(f"    Keyword: {ar.keyword}")

    # ----- Summary -----
    total_bl_ms = sum(r.latency_ms for r in baseline_results)
    total_apc_ms = sum(r.latency_ms for r in apc_results)
    total_bl_llm = sum(r.llm_calls for r in baseline_results)
    total_apc_llm = sum(r.llm_calls for r in apc_results)

    bl_expensive = total_bl_llm
    apc_expensive = apc.stats.cache_misses

    avg_steps_hit = []
    avg_steps_miss = []
    for ar in apc_results:
        if ar.cache_hit:
            avg_steps_hit.append(len(ar.plan.steps))
        else:
            avg_steps_miss.append(len(ar.plan.steps))

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
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

    if avg_steps_hit:
        print(f"  Avg steps (cache hit)  : {sum(avg_steps_hit)/len(avg_steps_hit):.1f}")
    if avg_steps_miss:
        print(f"  Avg steps (cache miss) : {sum(avg_steps_miss)/len(avg_steps_miss):.1f}")

    # Per cache-hit: LLM calls = 1 (keyword) + N (adapt) + 1 (synthesis) = N+2
    # Show the interleaved execution clearly
    print()
    print("  Interleaved execution detail (cache hits only):")
    for i, ar in enumerate(apc_results, 1):
        if ar.cache_hit:
            n = len(ar.plan.steps)
            print(
                f"    Q{i}: {n} steps → "
                f"1 keyword + {n} adapt + 1 synthesis = {ar.llm_calls} cheap LLM calls"
            )

    # ----- Answer comparison -----
    print()
    print("=" * 80)
    print("Answer Comparison")
    print("=" * 80)
    for i, (br, ar) in enumerate(zip(baseline_results, apc_results), 1):
        cache_str = "HIT" if ar.cache_hit else "MISS"
        print(f"\n  Q{i} [{cache_str}]: {br.query}")
        print(f"    Baseline : {br.final_answer[:100]}")
        print(f"    APC      : {ar.final_answer[:100]}")
    print()


if __name__ == "__main__":
    run_benchmark()
