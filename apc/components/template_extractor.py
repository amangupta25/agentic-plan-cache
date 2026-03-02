"""Extract a reusable plan template from an executed plan."""

from __future__ import annotations

import json
import re

from apc.llm.base import LLMProvider
from apc.models import Plan, PlanTemplate, TemplateStep
from apc.prompts import TEMPLATE_EXTRACTION_PROMPT


class TemplateExtractor:
    """Generalizes an executed plan into a reusable template.

    Uses a two-stage approach per the paper:
    1. Rule-based filter to strip verbose reasoning and query-specific args
    2. LLM generalization to produce parameter hints
    """

    _REASONING_PREFIXES = (
        "let me", "first,", "therefore,", "so,", "next,", "now,",
        "i will", "i'll", "i need", "we need", "note that", "recall",
        "since", "because", "given that", "as we",
    )

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def extract(self, plan: Plan) -> PlanTemplate:
        raw_steps = [
            {
                "index": s.index,
                "description": s.description,
                "tool_name": s.tool_name,
                "tool_args": s.tool_args,
                "result": s.result,
            }
            for s in plan.steps
        ]
        filtered_steps = self._rule_based_filter(raw_steps)

        executed_steps = json.dumps(filtered_steps, indent=2)
        prompt = TEMPLATE_EXTRACTION_PROMPT.format(
            source_query=plan.query,
            executed_steps=executed_steps,
        )
        response = self._llm.complete(prompt, temperature=0.0, max_tokens=1024)
        return self._parse(response.content, plan.query)

    def _rule_based_filter(self, steps: list[dict]) -> list[dict]:
        """Strip verbose reasoning and query-specific tool_args before LLM generalization."""
        cleaned = []
        for step in steps:
            desc = step.get("description", "")
            # Remove lines that look like reasoning
            lines = desc.splitlines()
            kept_lines = []
            for line in lines:
                stripped = line.strip().lower()
                if stripped.startswith("---"):
                    break
                if any(stripped.startswith(p) for p in self._REASONING_PREFIXES):
                    continue
                kept_lines.append(line)
            cleaned_desc = "\n".join(kept_lines).strip() or desc

            cleaned.append({
                "index": step.get("index"),
                "description": cleaned_desc,
                "tool_name": step.get("tool_name"),
                "result": step.get("result"),
            })
        return cleaned

    def _parse(self, content: str, source_query: str) -> PlanTemplate:
        content = content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                steps = [
                    TemplateStep(
                        index=s.get("index", i + 1),
                        description=s.get("description", ""),
                        tool_name=s.get("tool_name", ""),
                        parameter_hints=s.get("parameter_hints", {}),
                    )
                    for i, s in enumerate(data.get("steps", []))
                ]
                return PlanTemplate(
                    steps=steps,
                    source_query=source_query,
                    category=data.get("category", ""),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        return PlanTemplate(steps=[], source_query=source_query, category="unknown")
