"""Shared test fixtures — MockLLMProvider that returns canned JSON responses."""

from __future__ import annotations

import json
from typing import Any

import pytest

from apc.llm.base import LLMProvider, LLMResponse
from apc.tools.calculator import CalculatorTool
from apc.tools.web_search import WebSearchTool
from apc.tools.registry import ToolRegistry
from apc.cache.plan_cache import PlanCache


class MockLLMProvider(LLMProvider):
    """Mock LLM that returns pre-configured responses based on prompt content."""

    def __init__(self, name: str = "mock-model", responses: dict[str, str] | None = None):
        self._name = name
        self._responses = responses or {}
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self._call_count += 1
        # Check for keyword matches in the prompt
        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return LLMResponse(content=response, model=self._name)
        # Default empty response
        return LLMResponse(content="[]", model=self._name)


def make_keyword_response(keyword: str) -> str:
    return keyword


def make_plan_response(steps: list[dict[str, Any]]) -> str:
    return json.dumps(steps)


def make_template_response(category: str, steps: list[dict[str, Any]]) -> str:
    return json.dumps({"category": category, "steps": steps})


def make_step_response(step: dict[str, Any]) -> str:
    return json.dumps(step)


def make_answer_response(answer: str) -> str:
    return answer


# --- Canned responses for math queries ---

MATH_KEYWORD = make_keyword_response("square root addition")

MATH_PLAN = make_plan_response([
    {
        "index": 1,
        "description": "Calculate the square root of the number",
        "tool_name": "calculator",
        "tool_args": {"expression": "sqrt(144)"},
    },
    {
        "index": 2,
        "description": "Add the second number to the result",
        "tool_name": "calculator",
        "tool_args": {"expression": "12.0 + 12"},
    },
])

MATH_TEMPLATE = make_template_response("arithmetic", [
    {
        "index": 1,
        "description": "Calculate the square root of <number>",
        "tool_name": "calculator",
        "parameter_hints": {"expression": "sqrt(<number>)"},
    },
    {
        "index": 2,
        "description": "Add <addend> to the square root result",
        "tool_name": "calculator",
        "parameter_hints": {"expression": "<sqrt_result> + <addend>"},
    },
])

MATH_ADAPTED_STEP_1 = make_step_response({
    "index": 1,
    "description": "Calculate the square root of 256",
    "tool_name": "calculator",
    "tool_args": {"expression": "sqrt(256)"},
})

MATH_ADAPTED_STEP_2 = make_step_response({
    "index": 2,
    "description": "Add 20 to the result",
    "tool_name": "calculator",
    "tool_args": {"expression": "16.0 + 20"},
})

MATH_ANSWER = "The square root of 144 is 12, and adding 12 gives 24."
MATH_ANSWER_2 = "The square root of 256 is 16, and adding 20 gives 36."


@pytest.fixture
def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    return registry


@pytest.fixture
def plan_cache(tmp_path) -> PlanCache:
    return PlanCache(persist_path=tmp_path / "test_cache.json", max_size=128)


@pytest.fixture
def mock_expensive_llm() -> MockLLMProvider:
    """Expensive LLM used for planning from scratch."""
    return MockLLMProvider(
        name="expensive-mock",
        responses={
            "step-by-step plan": MATH_PLAN,
        },
    )


@pytest.fixture
def mock_cheap_llm() -> MockLLMProvider:
    """Cheap LLM used for keywords, adaptation, templates, and synthesis."""
    return MockLLMProvider(
        name="cheap-mock",
        responses={
            "higher-level intent": MATH_KEYWORD,
            "reuse this trace": MATH_TEMPLATE,
            "no prior steps": MATH_ADAPTED_STEP_1,
            "step 1": MATH_ADAPTED_STEP_2,
            "synthesize a clear final answer": MATH_ANSWER,
        },
    )
