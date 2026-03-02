#!/usr/bin/env python3
"""End-to-end demo of Agentic Plan Caching with math queries.

Usage:
    OPENAI_API_KEY=sk-... python demo/math_demo.py

Or set the env var in a .env file and source it first.
"""

from __future__ import annotations

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apc.agent import APCAgent
from apc.cache.plan_cache import PlanCache
from apc.llm.openai_provider import OpenAIProvider
from apc.tools.calculator import CalculatorTool
from apc.tools.web_search import WebSearchTool
from apc.tools.registry import ToolRegistry


def main() -> None:
    # --- Setup ---
    expensive_model = os.getenv("APC_EXPENSIVE_MODEL", "gpt-4o")
    cheap_model = os.getenv("APC_CHEAP_MODEL", "gpt-4o-mini")

    print(f"Expensive LLM: {expensive_model}")
    print(f"Cheap LLM:     {cheap_model}")
    print()

    expensive_llm = OpenAIProvider(model=expensive_model)
    cheap_llm = OpenAIProvider(model=cheap_model)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())

    cache = PlanCache(persist_path="cache.json", max_size=128)
    agent = APCAgent(
        expensive_llm=expensive_llm,
        cheap_llm=cheap_llm,
        tool_registry=registry,
        cache=cache,
    )

    # --- Queries ---
    queries = [
        "What is the square root of 144 plus 12?",
        "What is the square root of 256 plus 20?",
        "What is the square root of 81 plus 5?",
        "What is the square root of 625 plus 30?",
    ]

    print("=" * 60)
    print("Running APC Demo — similar math queries")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        result = agent.run(query)
        status = "CACHE HIT" if result.cache_hit else "CACHE MISS"
        print(f"  Status:    {status}")
        print(f"  Keyword:   {result.keyword}")
        print(f"  Answer:    {result.final_answer}")
        print(f"  Latency:   {result.latency_ms:.0f} ms")
        print(f"  LLM calls: {result.llm_calls}")
        print(f"  Plan src:  {result.plan.source}")

    # --- Summary ---
    stats = agent.stats
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total queries:  {stats.total_queries}")
    print(f"  Cache hits:     {stats.cache_hits}")
    print(f"  Cache misses:   {stats.cache_misses}")
    print(f"  Hit rate:       {stats.hit_rate:.0%}")
    print(f"  Avg latency:    {stats.avg_latency_ms:.0f} ms")
    print(f"  Cache size:     {agent.cache.size} templates")
    print(f"  Total LLM calls: {stats.total_llm_calls}")

    # Estimate savings: miss = 4 LLM calls (1 expensive), hit = N+2 LLM calls (0 expensive)
    expensive_calls_actual = stats.cache_misses  # 1 expensive call per miss
    expensive_calls_without_cache = stats.total_queries  # would be 1 per query
    if expensive_calls_without_cache > 0:
        savings = 1 - (expensive_calls_actual / expensive_calls_without_cache)
        print(f"  Expensive LLM savings: ~{savings:.0%}")


if __name__ == "__main__":
    main()
