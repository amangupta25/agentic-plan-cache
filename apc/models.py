"""Pydantic data models for Agentic Plan Caching."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in a concrete execution plan."""

    index: int
    description: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: str | None = None


class Plan(BaseModel):
    """A concrete plan generated for a specific query."""

    query: str
    steps: list[PlanStep]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["planner", "adapted"] = "planner"


class TemplateStep(BaseModel):
    """A generalized step in a reusable plan template."""

    index: int
    description: str
    tool_name: str
    parameter_hints: dict[str, str] = Field(default_factory=dict)


class PlanTemplate(BaseModel):
    """A reusable plan template extracted from a successful execution."""

    steps: list[TemplateStep]
    source_query: str
    category: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0


class CacheEntry(BaseModel):
    """A cached plan template keyed by a single keyword."""

    keyword: str
    template: PlanTemplate


class ExecutionResult(BaseModel):
    """Result of running a query through the APC agent."""

    query: str
    plan: Plan
    final_answer: str
    success: bool
    cache_hit: bool
    latency_ms: float
    llm_calls: int
    keyword: str = ""
