"""Calculator tool for safe math expression evaluation."""

from __future__ import annotations

import math
from typing import Any

from .base import Tool

# Allowed names for safe eval
_SAFE_NAMES: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}


class CalculatorTool(Tool):
    """Evaluates mathematical expressions safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, log, trig functions, etc."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs: Any) -> str:
        expression = kwargs.get("expression", "")
        if not expression:
            return "Error: no expression provided"
        try:
            result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"
