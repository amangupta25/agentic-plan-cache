#!/usr/bin/env python3
"""Benchmark: multi-step workflows (2-5 steps each) — APC vs Baseline.

Tests the interleaved adapt→execute loop on diverse real-world workflows
that chain web_search and calculator across multiple domains.

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
# Diverse multi-step queries grouped by domain
# ---------------------------------------------------------------------------
QUERIES = [
    # --- Group 1: Physics — search constant → compute with it ---
    "Find the speed of light in m/s, then calculate how far light travels in 5 seconds.",
    "Find the speed of light in m/s, then calculate how far light travels in 10 seconds.",
    "Find the speed of light in m/s, then calculate how far light travels in 3 seconds.",

    # --- Group 2: Geography — search fact → do math ---
    "Find the population of India and the population of the United States, then compute how many times larger India's population is.",
    "Find the population of China and the population of the United States, then compute how many times larger China's population is.",

    # --- Group 3: Finance — search rate → convert currency ---
    "Look up the USD to EUR exchange rate, then convert $5,000 to euros.",
    "Look up the USD to GBP exchange rate, then convert $5,000 to British pounds.",
    "Look up the USD to JPY exchange rate, then convert $5,000 to Japanese yen.",

    # --- Group 4: Health/Nutrition — search calories → compute daily totals ---
    "Find the calories in a banana and in an egg. If I eat 3 bananas and 2 eggs, how many total calories is that?",
    "Find the calories in a banana and in rice. If I eat 2 bananas and 1 cup of rice, how many total calories is that?",

    # --- Group 5: Unit conversion — search conversion factor → calculate ---
    "Find how many miles are in a kilometer, then convert a 42.195 km marathon to miles.",
    "Find how many pounds are in a kilogram, then convert 75 kg to pounds.",
    "Find how many liters are in a gallon, then convert 10 gallons to liters.",

    # --- Group 6: Geography comparison — search two facts → compute ---
    "Find the height of Mount Everest and the depth of the Mariana Trench. What is their combined vertical span?",
    "Find the length of the Amazon River and the Nile River. How much longer is the Nile?",

    # --- Group 7: Finance — search price → compute portfolio ---
    "Look up the current price of gold per ounce, then calculate the value of 5 ounces.",
    "Look up the current price of gold per ounce, then calculate the value of 12 ounces.",

    # --- Group 8: Everyday math — search info → compute tip/cost ---
    "Find the US minimum wage, then calculate the weekly earnings for a 40-hour work week before tax.",
    "Find the US minimum wage, then calculate the annual earnings for a 40-hour work week over 52 weeks.",

    # --- Group 9: Math — pure compute chains ---
    "What is the square root of 144 plus 12?",
    "What is the square root of 256 plus 20?",
    "What is the square root of 81 plus 5?",
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

    print("=" * 85)
    print("Multi-Step Workflow Benchmark — APC vs Baseline (Diverse Queries)")
    print("=" * 85)
    print(f"  Expensive LLM : {expensive_model}")
    print(f"  Cheap LLM     : {cheap_model}")
    print(f"  Queries       : {len(QUERIES)}")
    print(f"  Domains       : physics, geography, finance, health, unit conversion, everyday, math")
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
    print("=" * 85)
    print("Per-Query Results")
    print("=" * 85)

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

    # Unique keywords
    unique_keywords = set(ar.keyword for ar in apc_results)

    print()
    print("=" * 85)
    print("Summary")
    print("=" * 85)
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
    print(f"  Unique keywords  : {len(unique_keywords)}")

    if avg_steps_hit:
        print(f"  Avg steps (cache hit)  : {sum(avg_steps_hit)/len(avg_steps_hit):.1f}")
    if avg_steps_miss:
        print(f"  Avg steps (cache miss) : {sum(avg_steps_miss)/len(avg_steps_miss):.1f}")

    # Interleaved detail
    print()
    print("  Interleaved execution detail (cache hits only):")
    for i, ar in enumerate(apc_results, 1):
        if ar.cache_hit:
            n = len(ar.plan.steps)
            print(
                f"    Q{i:>2}: {n} steps → "
                f"1 keyword + {n} adapt + 1 synthesis = {ar.llm_calls} cheap LLM calls"
            )

    # Keyword clustering
    print()
    print("  Keyword → Query mapping:")
    keyword_groups: dict[str, list[int]] = {}
    for i, ar in enumerate(apc_results, 1):
        keyword_groups.setdefault(ar.keyword, []).append(i)
    for kw, qnums in sorted(keyword_groups.items(), key=lambda x: -len(x[1])):
        hit_count = sum(1 for q in qnums if apc_results[q - 1].cache_hit)
        print(f"    \"{kw}\" → Q{qnums} ({hit_count} hits)")

    # ----- Answer comparison -----
    print()
    print("=" * 85)
    print("Answer Comparison (first 100 chars)")
    print("=" * 85)
    for i, (br, ar) in enumerate(zip(baseline_results, apc_results), 1):
        cache_str = "HIT" if ar.cache_hit else "MISS"
        print(f"\n  Q{i} [{cache_str}]: {br.query}")
        print(f"    Baseline : {br.final_answer[:100]}")
        print(f"    APC      : {ar.final_answer[:100]}")
    print()


if __name__ == "__main__":
    run_benchmark()
