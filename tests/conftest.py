from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvault.audit import AuditSink
from aegisvault.evaluators import ScopeEvaluator
from aegisvault.exceptions import EvaluatorTimeoutError, MalformedEvaluatorResponseError
from aegisvault.policy.models import DomainPolicy
from aegisvault.types import EvaluationContext, GateDecision, GateType, Verdict


class FakeEvaluator(ScopeEvaluator):
    def __init__(self, decisions: list[GateDecision] | None = None, exc: Exception | None = None) -> None:
        self.decisions = decisions or []
        self.exc = exc
        self.calls: list[tuple[str, GateType]] = []

    def evaluate(
        self,
        text: str,
        policy: DomainPolicy,
        gate_type: GateType,
        context: EvaluationContext | None = None,
    ) -> GateDecision:
        self.calls.append((text, gate_type))
        if self.exc is not None:
            raise self.exc
        if self.decisions:
            decision = self.decisions.pop(0)
            decision.gate = gate_type
            return decision
        return decision(Verdict.ALLOW, 0.95, gate_type)


class MemoryAuditSink(AuditSink):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def decision(verdict: Verdict, confidence: float | None, gate: GateType, reason: str = "fake") -> GateDecision:
    return GateDecision(
        verdict=verdict,
        confidence=confidence,
        reason=reason,
        gate=gate,
        evaluator="fake",
        latency_ms=1.0,
    )


@pytest.fixture
def policy_dict(tmp_path: Path) -> dict[str, Any]:
    return {
        "version": "1.0",
        "application": {"name": "test-app", "description": "Test assistant"},
        "purpose": "Help users with support requests.",
        "allowed_topics": ["support", "orders"],
        "blocked_topics": ["programming"],
        "gates": {
            "request": {
                "enabled": True,
                "allow_threshold": 0.8,
                "block_threshold": 0.8,
                "low_confidence_action": "clarify",
            },
            "response": {
                "enabled": True,
                "allow_threshold": 0.8,
                "block_threshold": 0.8,
                "low_confidence_action": "block",
            },
        },
        "evaluator": {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
            "timeout_seconds": 1,
            "temperature": 0,
        },
        "fallback": {"evaluator_failure_action": "block", "malformed_output_action": "block"},
        "checks": {
            "max_request_chars": 100,
            "max_response_chars": 100,
            "blocked_phrases": ["FORBIDDEN PHRASE"],
            "blocked_keywords": ["blockedword"],
            "keyword_case_insensitive": True,
        },
        "audit": {
            "enabled": True,
            "output_path": str(tmp_path / "audit.jsonl"),
            "include_request_text": True,
            "include_response_text": True,
        },
    }


@pytest.fixture
def policy(policy_dict: dict[str, Any]) -> DomainPolicy:
    return DomainPolicy.model_validate(policy_dict)


@pytest.fixture
def malformed_error() -> MalformedEvaluatorResponseError:
    return MalformedEvaluatorResponseError("bad evaluator output")


@pytest.fixture
def timeout_error() -> EvaluatorTimeoutError:
    return EvaluatorTimeoutError("timed out")
