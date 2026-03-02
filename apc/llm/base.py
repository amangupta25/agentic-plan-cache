"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name used by this provider."""
        ...
