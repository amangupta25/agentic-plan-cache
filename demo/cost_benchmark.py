#!/usr/bin/env python3
"""Cost benchmark: does APC actually save money?

Wraps real LLM providers to track per-call token counts,
then computes dollar costs for multiple model pairs.

Usage:
    source .env && .venv/bin/python demo/cost_benchmark.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apc.agent import APCAgent
from apc.baseline import BaselineAgent
from apc.cache.plan_cache import PlanCache
from apc.llm.base import LLMProvider, LLMResponse
from apc.llm.openai_provider import OpenAIProvider
from apc.tools.calculator import CalculatorTool
from apc.tools.registry import ToolRegistry
from apc.tools.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# Pricing per 1M tokens (USD) — as of early 2025
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":           {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4.1":          {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":     {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":     {"input": 0.10,  "output": 0.40},
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
}

# Model pairs to analyze: (expensive, cheap)
MODEL_PAIRS = [
    ("gpt-4o",           "gpt-4o-mini"),
    ("gpt-4.1",          "gpt-4.1-mini"),
    ("gpt-4.1",          "gpt-4.1-nano"),
    ("claude-sonnet-4-6", "claude-haiku-4-5"),
]


@dataclass
class TokenLog:
    """Tracks tokens per call with role labels."""
    calls: list[dict] = field(default_factory=list)

    def record(self, role: str, model: str, prompt_tokens: int, completion_tokens: int):
        self.calls.append({
            "role": role,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c["prompt_tokens"] for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c["completion_tokens"] for c in self.calls)

    def tokens_by_model(self, model: str) -> tuple[int, int]:
        pt = sum(c["prompt_tokens"] for c in self.calls if c["model"] == model)
        ct = sum(c["completion_tokens"] for c in self.calls if c["model"] == model)
        return pt, ct

    def cost(self, pricing: dict[str, dict[str, float]]) -> float:
        total = 0.0
        for call in self.calls:
            model = call["model"]
            if model not in pricing:
                # Try to match partial name
                for k in pricing:
                    if k in model or model in k:
                        model = k
                        break
                else:
                    continue
            p = pricing[model]
            total += (call["prompt_tokens"] / 1_000_000) * p["input"]
            total += (call["completion_tokens"] / 1_000_000) * p["output"]
        return total


class TrackingProvider(LLMProvider):
    """Wraps a real LLM provider and logs token usage."""

    def __init__(self, inner: LLMProvider, log: TokenLog, role: str = ""):
        self._inner = inner
        self._log = log
        self._role = role

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        resp = self._inner.complete(prompt, **kwargs)
        pt = resp.usage.get("prompt_tokens", 0)
        ct = resp.usage.get("completion_tokens", 0)
        self._log.record(self._role, self._inner.model_name, pt, ct)
        return resp


QUERIES = [
    # Group 1: sqrt + addition (3 similar)
    "What is the square root of 144 plus 12?",
    "What is the square root of 256 plus 20?",
    "What is the square root of 81 plus 5?",
    # Group 2: search + compute (3 similar)
    "Find the speed of light in m/s, then calculate how far light travels in 5 seconds.",
    "Find the speed of light in m/s, then calculate how far light travels in 10 seconds.",
    "Find the speed of light in m/s, then calculate how far light travels in 3 seconds.",
    # Group 3: search + convert (2 similar)
    "Find the distance from Earth to the Sun in km, then convert it to miles by dividing by 1.609.",
    "Find the distance from Earth to the Sun in km, then convert it to meters by multiplying by 1000.",
    # Group 4: unique
    "Calculate 2 raised to the power of 10 then subtract 24.",
]


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool())
    return reg


def run() -> None:
    expensive_model = os.getenv("APC_EXPENSIVE_MODEL", "gpt-4o")
    cheap_model = os.getenv("APC_CHEAP_MODEL", "gpt-4o-mini")

    print("=" * 80)
    print("Cost Analysis Benchmark — Does APC Actually Save Money?")
    print("=" * 80)
    print(f"  Expensive model : {expensive_model}")
    print(f"  Cheap model     : {cheap_model}")
    print(f"  Queries         : {len(QUERIES)}")
    print()

    raw_expensive = OpenAIProvider(model=expensive_model)
    raw_cheap = OpenAIProvider(model=cheap_model)

    # --- Baseline: all calls use expensive model ---
    bl_log = TokenLog()
    bl_expensive = TrackingProvider(raw_expensive, bl_log, role="expensive")

    print("Running BASELINE ...")
    baseline = BaselineAgent(llm=bl_expensive, tool_registry=build_registry())
    for q in QUERIES:
        baseline.run(q)

    # --- APC: expensive for planning, cheap for everything else ---
    apc_log = TokenLog()
    apc_expensive = TrackingProvider(raw_expensive, apc_log, role="expensive")
    apc_cheap = TrackingProvider(raw_cheap, apc_log, role="cheap")

    print("Running APC ...")
    apc = APCAgent(
        expensive_llm=apc_expensive,
        cheap_llm=apc_cheap,
        tool_registry=build_registry(),
        cache=PlanCache(max_size=128),
    )
    for q in QUERIES:
        apc.run(q)

    # --- Token summary ---
    print()
    print("=" * 80)
    print("Token Usage")
    print("=" * 80)

    bl_pt = bl_log.total_prompt_tokens
    bl_ct = bl_log.total_completion_tokens
    apc_pt = apc_log.total_prompt_tokens
    apc_ct = apc_log.total_completion_tokens

    apc_exp_pt, apc_exp_ct = apc_log.tokens_by_model(expensive_model)
    apc_chp_pt, apc_chp_ct = apc_log.tokens_by_model(cheap_model)

    print(f"\n  {'':30} {'Prompt':>12} {'Completion':>12} {'Total':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Baseline (all expensive)':<30} {bl_pt:>12,} {bl_ct:>12,} {bl_pt+bl_ct:>12,}")
    print(f"  {'APC total':<30} {apc_pt:>12,} {apc_ct:>12,} {apc_pt+apc_ct:>12,}")
    print(f"    {'└ expensive calls':<28} {apc_exp_pt:>12,} {apc_exp_ct:>12,} {apc_exp_pt+apc_exp_ct:>12,}")
    print(f"    {'└ cheap calls':<28} {apc_chp_pt:>12,} {apc_chp_ct:>12,} {apc_chp_pt+apc_chp_ct:>12,}")

    token_ratio = (apc_pt + apc_ct) / (bl_pt + bl_ct) if (bl_pt + bl_ct) else 0
    print(f"\n  APC uses {token_ratio:.1f}x the total tokens of baseline")

    # --- Cost comparison across model pairs ---
    print()
    print("=" * 80)
    print("Cost Comparison Across Model Pairs")
    print("=" * 80)
    print(f"\n  Using actual token counts from the run above,")
    print(f"  projected onto different pricing tiers.\n")

    print(f"  {'Expensive / Cheap':<35} {'Baseline':>10} {'APC':>10} {'Saving':>10} {'Saved?':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for exp_model, chp_model in MODEL_PAIRS:
        if exp_model not in MODEL_PRICING or chp_model not in MODEL_PRICING:
            continue

        # Baseline cost: all tokens priced at expensive rate
        bl_cost = (
            (bl_pt / 1_000_000) * MODEL_PRICING[exp_model]["input"]
            + (bl_ct / 1_000_000) * MODEL_PRICING[exp_model]["output"]
        )

        # APC cost: expensive tokens at expensive rate, cheap tokens at cheap rate
        apc_cost = (
            (apc_exp_pt / 1_000_000) * MODEL_PRICING[exp_model]["input"]
            + (apc_exp_ct / 1_000_000) * MODEL_PRICING[exp_model]["output"]
            + (apc_chp_pt / 1_000_000) * MODEL_PRICING[chp_model]["input"]
            + (apc_chp_ct / 1_000_000) * MODEL_PRICING[chp_model]["output"]
        )

        saving_pct = (1 - apc_cost / bl_cost) * 100 if bl_cost else 0
        verdict = "YES" if saving_pct > 0 else "NO"

        pair_label = f"{exp_model} / {chp_model}"
        print(
            f"  {pair_label:<35} ${bl_cost:>8.4f} ${apc_cost:>8.4f} "
            f"{saving_pct:>+9.1f}% {verdict:>8}"
        )

    # --- Per-call breakdown ---
    print()
    print("=" * 80)
    print("Per-Call Detail (APC)")
    print("=" * 80)

    exp_calls = [c for c in apc_log.calls if c["role"] == "expensive"]
    chp_calls = [c for c in apc_log.calls if c["role"] == "cheap"]

    print(f"\n  Expensive LLM calls: {len(exp_calls)}")
    for i, c in enumerate(exp_calls, 1):
        print(f"    {i}. {c['prompt_tokens']:>5} prompt + {c['completion_tokens']:>5} completion")

    print(f"\n  Cheap LLM calls: {len(chp_calls)}")
    for i, c in enumerate(chp_calls, 1):
        print(f"    {i:>2}. {c['prompt_tokens']:>5} prompt + {c['completion_tokens']:>5} completion")

    print()


if __name__ == "__main__":
    run()
