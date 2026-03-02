"""Execute plan steps and synthesize a final answer."""

from __future__ import annotations

from apc.llm.base import LLMProvider
from apc.models import Plan
from apc.prompts import ANSWER_SYNTHESIS_PROMPT
from apc.tools.registry import ToolRegistry


class Actor:
    """Executes plan steps via the tool registry and synthesizes a final answer."""

    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry) -> None:
        self._llm = llm
        self._tool_registry = tool_registry

    def execute(self, plan: Plan) -> str:
        """Execute all plan steps, updating each step's status and result."""
        for step in plan.steps:
            step.status = "running"
            result = self._tool_registry.execute(step.tool_name, **step.tool_args)
            step.result = result
            step.status = "completed" if not result.startswith("Error") else "failed"

        return self.synthesize(plan)

    def synthesize(self, plan: Plan) -> str:
        """Synthesize a final answer from executed plan step results."""
        step_results = "\n".join(
            f"Step {s.index}: {s.description}\n  Tool: {s.tool_name}\n  Result: {s.result}"
            for s in plan.steps
        )
        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            query=plan.query,
            step_results=step_results,
        )
        response = self._llm.complete(prompt, temperature=0.0, max_tokens=512)
        return response.content.strip()
