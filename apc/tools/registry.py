"""Tool registry for managing available tools."""

from __future__ import annotations

from typing import Any

from .base import Tool


class ToolRegistry:
    """Registry that maps tool names to Tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {name}: {e}"

    def list_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
