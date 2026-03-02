"""APCAgent — main orchestrator for Agentic Plan Caching."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from apc.cache.plan_cache import PlanCache
from apc.components.actor import Actor
from apc.components.keyword_extractor import KeywordExtractor
from apc.components.planner import Planner
from apc.components.template_adapter import TemplateAdapter
from apc.components.template_extractor import TemplateExtractor
from apc.llm.base import LLMProvider
from apc.models import ExecutionResult, Plan, PlanTemplate
from apc.tools.registry import ToolRegistry


@dataclass
class AgentStats:
    """Running statistics for the APC agent."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_llm_calls: int = 0
    _latencies: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0


class APCAgent:
    """Agentic Plan Caching agent.

    On cache miss: uses expensive LLM to plan from scratch, then caches the template.
    On cache hit: uses interleaved adapt→execute loop with cheap LLM.
    """

    def __init__(
        self,
        expensive_llm: LLMProvider,
        cheap_llm: LLMProvider,
        tool_registry: ToolRegistry,
        cache: PlanCache | None = None,
    ) -> None:
        self._cache = cache or PlanCache()
        self._keyword_extractor = KeywordExtractor(cheap_llm)
        self._planner = Planner(expensive_llm, tool_registry)
        self._template_extractor = TemplateExtractor(cheap_llm)
        self._template_adapter = TemplateAdapter(cheap_llm, tool_registry)
        self._actor = Actor(cheap_llm, tool_registry)
        self._tool_registry = tool_registry
        self._stats = AgentStats()

    @property
    def stats(self) -> AgentStats:
        return self._stats

    @property
    def cache(self) -> PlanCache:
        return self._cache

    def run(self, query: str) -> ExecutionResult:
        start = time.perf_counter()
        llm_calls = 0

        # 1. Extract keyword (cheap LLM)
        keyword = self._keyword_extractor.extract(query)
        llm_calls += 1

        # 2. Cache lookup
        template = self._cache.lookup(keyword)
        cache_hit = template is not None

        if cache_hit:
            # 3a. HIT — interleaved adapt→execute loop
            plan, adapt_calls = self._interleaved_execute(template, query)
            llm_calls += adapt_calls
            # Synthesize final answer (cheap LLM)
            final_answer = self._actor.synthesize(plan)
            llm_calls += 1
        else:
            # 3b. MISS — generate plan from scratch (expensive LLM)
            plan = self._planner.plan(query)
            llm_calls += 1

            # 4. Execute plan (actor + tools) + synthesize answer (cheap LLM)
            final_answer = self._actor.execute(plan)
            llm_calls += 1

        success = any(s.status == "completed" for s in plan.steps)

        # 5. On miss + success: extract template and cache it
        if not cache_hit and success:
            new_template = self._template_extractor.extract(plan)
            llm_calls += 1
            if new_template.steps:
                self._cache.store(keyword, new_template)

        elapsed_ms = (time.perf_counter() - start) * 1000

        self._update_stats(llm_calls, elapsed_ms, cache_hit)

        return ExecutionResult(
            query=query,
            plan=plan,
            final_answer=final_answer,
            success=success,
            cache_hit=cache_hit,
            latency_ms=round(elapsed_ms, 2),
            llm_calls=llm_calls,
            keyword=keyword,
        )

    def _interleaved_execute(
        self, template: PlanTemplate, query: str
    ) -> tuple[Plan, int]:
        """Adapt and execute template steps one at a time (paper's interleaved loop).

        Each step is adapted with prior results as context, then immediately executed.
        Returns the completed Plan and the number of LLM calls made (N adapt calls).
        """
        executed_steps = []
        llm_calls = 0

        for template_step in template.steps:
            # Adapt this step using prior results as context
            concrete_step = self._template_adapter.adapt_step(
                template_step, query, executed_steps
            )
            llm_calls += 1

            # Execute immediately
            result = self._tool_registry.execute(
                concrete_step.tool_name, **concrete_step.tool_args
            )
            concrete_step.result = result
            concrete_step.status = (
                "completed" if not result.startswith("Error") else "failed"
            )
            executed_steps.append(concrete_step)

        plan = Plan(query=query, steps=executed_steps, source="adapted")
        return plan, llm_calls

    def _update_stats(
        self, llm_calls: int, elapsed_ms: float, cache_hit: bool
    ) -> None:
        self._stats.total_queries += 1
        self._stats.total_llm_calls += llm_calls
        self._stats._latencies.append(elapsed_ms)
        if cache_hit:
            self._stats.cache_hits += 1
        else:
            self._stats.cache_misses += 1
