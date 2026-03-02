"""Generate a plan from scratch using an expensive LLM."""

from __future__ import annotations

import json
import re

from apc.llm.base import LLMProvider
from apc.models import Plan, PlanStep
from apc.prompts import PLAN_GENERATION_PROMPT
from apc.tools.registry import ToolRegistry


class Planner:
    """Generates a concrete plan for a query using an expensive LLM."""

    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry) -> None:
        self._llm = llm
        self._tool_registry = tool_registry

    def plan(self, query: str) -> Plan:
        tool_schemas = json.dumps(self._tool_registry.list_schemas(), indent=2)
        prompt = PLAN_GENERATION_PROMPT.format(
            query=query,
            tool_schemas=tool_schemas,
        )
        response = self._llm.complete(prompt, temperature=0.0, max_tokens=1024)
        return self._parse(response.content, query)

    def _parse(self, content: str, query: str) -> Plan:
        content = content.strip()
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                steps = [
                    PlanStep(
                        index=s.get("index", i + 1),
                        description=str(s.get("description", "")),
                        tool_name=str(s.get("tool_name", "")),
                        tool_args=s.get("tool_args", {}),
                    )
                    for i, s in enumerate(data)
                ]
                return Plan(query=query, steps=steps, source="planner")
            except (json.JSONDecodeError, KeyError):
                pass

        return Plan(query=query, steps=[], source="planner")
