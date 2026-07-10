"""Scope evaluator interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegisvault.policy.models import DomainPolicy
from aegisvault.types import EvaluationContext, GateDecision, GateType


class ScopeEvaluator(ABC):
    """Abstract interface for domain scope evaluators."""

    @abstractmethod
    def evaluate(
        self,
        text: str,
        policy: DomainPolicy,
        gate_type: GateType,
        context: EvaluationContext | None = None,
    ) -> GateDecision:
        """Evaluate whether text belongs within a policy's declared domain."""
