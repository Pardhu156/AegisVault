"""Shared typed objects used across AegisVault."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Possible gate verdicts."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    CLARIFY = "CLARIFY"
    REPLACE = "REPLACE"


class GateType(str, Enum):
    """Supported gate types."""

    REQUEST = "request"
    RESPONSE = "response"


class TerminatedBy(str, Enum):
    """Component that produced the final response."""

    LAYER0 = "LAYER0"
    REQUEST_GATE = "REQUEST_GATE"
    RESPONSE_GATE = "RESPONSE_GATE"
    APPLICATION = "APPLICATION"


@dataclass(slots=True)
class EvaluationContext:
    """Optional context passed to a scope evaluator."""

    request_text: str | None = None
    response_text: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateDecision:
    """Decision returned by a request or response gate."""

    verdict: Verdict
    confidence: float | None
    reason: str
    gate: GateType
    evaluator: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuardResult:
    """Structured result returned by a wrapped callable."""

    final_response: str
    request_decision: GateDecision | None
    response_decision: GateDecision | None
    application_called: bool
    request_accepted: bool
    response_accepted: bool | None
    was_modified: bool
    terminated_by: TerminatedBy
    original_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
