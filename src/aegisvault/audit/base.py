"""Audit sink interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AuditSink(ABC):
    """Abstract destination for structured audit events."""

    @abstractmethod
    def record(self, event: dict[str, Any]) -> None:
        """Persist one audit event."""


class NullAuditSink(AuditSink):
    """Audit sink that intentionally drops events."""

    def record(self, event: dict[str, Any]) -> None:
        return None
