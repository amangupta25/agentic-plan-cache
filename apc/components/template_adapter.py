"""Adapt a single template step to a new query using a cheap LLM (interleaved execution)."""

from __future__ import annotations

import json
import re

from apc.llm.base import LLMProvider
from apc.models import PlanStep, TemplateStep
from apc.prompts import STEP_ADAPTATION_PROMPT
from apc.tools.registry import ToolRegistry


class TemplateAdapter:
    """Adapts cached template steps one at a time, with prior results as context."""

    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry) -> None:
        self._llm = llm
        self._tool_registry = tool_registry

    def adapt_step(
        self,
        template_step: TemplateStep,
        query: str,
        prior_results: list[PlanStep],
    ) -> PlanStep:
        """Adapt a single template step, given prior executed steps as context."""
        template_step_str = json.dumps(
            {
                "index": template_step.index,
                "description": template_step.description,
                "tool_name": template_step.tool_name,
                "parameter_hints": template_step.parameter_hints,
            },
            indent=2,
        )
        tool_schemas = json.dumps(self._tool_registry.list_schemas(), indent=2)
        prior_results_str = self._format_prior_results(prior_results)

        prompt = STEP_ADAPTATION_PROMPT.format(
            template_step=template_step_str,
            query=query,
            tool_schemas=tool_schemas,
            prior_results=prior_results_str,
        )
        response = self._llm.complete(prompt, temperature=0.0, max_tokens=512)
        return self._parse(response.content, template_step)

    def _format_prior_results(self, prior_results: list[PlanStep]) -> str:
        if not prior_results:
            return "(no prior steps)"
        lines = []
        for step in prior_results:
            lines.append(
                f"Step {step.index}: {step.description}\n"
                f"  Tool: {step.tool_name}\n"
                f"  Result: {step.result}"
            )
        return "\n".join(lines)

    def _parse(self, content: str, template_step: TemplateStep) -> PlanStep:
        content = content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return PlanStep(
                    index=data.get("index", template_step.index),
                    description=str(data.get("description", template_step.description)),
                    tool_name=str(data.get("tool_name", template_step.tool_name)),
                    tool_args=data.get("tool_args", {}),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        return PlanStep(
            index=template_step.index,
            description=template_step.description,
            tool_name=template_step.tool_name,
            tool_args={},
        )
