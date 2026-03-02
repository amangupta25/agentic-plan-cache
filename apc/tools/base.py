"""Abstract tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for tools that the agent can execute."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON-Schema-style description of expected keyword arguments."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result."""
        ...
