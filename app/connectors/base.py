from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConnectorError(RuntimeError):
    """Raised when a connector operation fails."""


class Connector(ABC):
    """Base connector interface for model backends."""

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        """Check backend availability and return status metadata."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names for this backend."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        """Run chat completion and return assistant message content."""
