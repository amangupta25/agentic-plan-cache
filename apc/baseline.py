"""BaselineAgent — no-cache baseline that always plans from scratch with expensive LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from apc.components.actor import Actor
from apc.components.planner import Planner
from apc.llm.base import LLMProvider
from apc.models import ExecutionResult
from apc.tools.registry import ToolRegistry


@dataclass
class BaselineStats:
    """Running statistics for the baseline agent."""

    total_queries: int = 0
    total_llm_calls: int = 0
    _latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0


class BaselineAgent:
    """No-cache baseline agent.

    Uses a single expensive LLM for everything: planning from scratch and
    synthesizing the final answer.  No keyword extraction, no template
    caching — this is the "without APC" scenario.
    """

    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry) -> None:
        self._planner = Planner(llm, tool_registry)
        self._actor = Actor(llm, tool_registry)
        self._stats = BaselineStats()

    @property
    def stats(self) -> BaselineStats:
        return self._stats

    def run(self, query: str) -> ExecutionResult:
        start = time.perf_counter()
        llm_calls = 0

        # 1. Plan from scratch (expensive LLM)
        plan = self._planner.plan(query)
        llm_calls += 1

        # 2. Execute plan steps + synthesize answer (expensive LLM)
        final_answer = self._actor.execute(plan)
        llm_calls += 1

        success = any(s.status == "completed" for s in plan.steps)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Update stats
        self._stats.total_queries += 1
        self._stats.total_llm_calls += llm_calls
        self._stats._latencies.append(elapsed_ms)

        return ExecutionResult(
            query=query,
            plan=plan,
            final_answer=final_answer,
            success=success,
            cache_hit=False,
            latency_ms=round(elapsed_ms, 2),
            llm_calls=llm_calls,
            keyword="",
        )
