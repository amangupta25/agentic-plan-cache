"""Simulated web search tool for demo purposes."""

from __future__ import annotations

from typing import Any

from .base import Tool

_CANNED_RESULTS: dict[str, str] = {
    "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "population of earth": "The world population is approximately 8.1 billion people as of 2024.",
    "distance earth to sun": "The average distance from Earth to the Sun is about 149.6 million kilometers (1 AU).",
    "boiling point of water": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
    "gravitational constant": "The gravitational constant G is approximately 6.674 × 10^-11 N⋅m²/kg².",
    "planck constant": "The Planck constant h is approximately 6.626 × 10^-34 J⋅s.",
    "avogadro number": "Avogadro's number is approximately 6.022 × 10^23 mol^-1.",
    "pi digits": "Pi (π) = 3.14159265358979323846...",
    "euler number": "Euler's number (e) = 2.71828182845904523536...",
}


class WebSearchTool(Tool):
    """Simulated web search that returns canned results for demo purposes."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for factual information. Returns relevant snippets."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").lower().strip()
        if not query:
            return "Error: no query provided"

        for key, result in _CANNED_RESULTS.items():
            if key in query or query in key:
                return result

        return f"No results found for '{query}'. Try a different search term."
