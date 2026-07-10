"""Request gate implementation."""

from __future__ import annotations

import time

from aegisvault.checks import check_request_text
from aegisvault.evaluators.base import ScopeEvaluator
from aegisvault.exceptions import EvaluatorError, MalformedEvaluatorResponseError
from aegisvault.policy.models import DomainPolicy, FallbackAction, LowConfidenceAction
from aegisvault.types import EvaluationContext, GateDecision, GateType, Verdict


class RequestGate:
    """Determines whether a user request may reach the protected callable."""

    def __init__(self, policy: DomainPolicy, evaluator: ScopeEvaluator) -> None:
        self.policy = policy
        self.evaluator = evaluator

    def evaluate(self, text: str, context: EvaluationContext | None = None) -> GateDecision:
        """Evaluate request text and return the final threshold-adjusted decision."""

        if not self.policy.gates.request.enabled:
            return GateDecision(
                verdict=Verdict.ALLOW,
                confidence=None,
                reason="Request gate is disabled by policy.",
                gate=GateType.REQUEST,
                evaluator="disabled",
                latency_ms=0.0,
            )

        deterministic = check_request_text(text, self.policy)
        if deterministic is not None:
            return deterministic

        started = time.perf_counter()
        try:
            raw = self.evaluator.evaluate(text, self.policy, GateType.REQUEST, context)
        except MalformedEvaluatorResponseError as exc:
            return self._fallback(self.policy.fallback.malformed_output_action, str(exc), started)
        except EvaluatorError as exc:
            return self._fallback(self.policy.fallback.evaluator_failure_action, str(exc), started)

        return self._apply_thresholds(raw)

    def _apply_thresholds(self, decision: GateDecision) -> GateDecision:
        config = self.policy.gates.request
        confidence = decision.confidence
        if decision.verdict == Verdict.ALLOW and confidence is not None and confidence >= config.allow_threshold:
            return decision
        if decision.verdict == Verdict.BLOCK and confidence is not None and confidence >= config.block_threshold:
            return decision

        return GateDecision(
            verdict=_request_low_confidence_verdict(config.low_confidence_action),
            confidence=confidence,
            reason=f"Evaluator confidence did not meet request thresholds. Original reason: {decision.reason}",
            gate=GateType.REQUEST,
            evaluator=decision.evaluator,
            latency_ms=decision.latency_ms,
            metadata={**decision.metadata, "original_verdict": decision.verdict.value},
        )

    def _fallback(self, action: FallbackAction, reason: str, started: float) -> GateDecision:
        return GateDecision(
            verdict=_request_fallback_verdict(action),
            confidence=None,
            reason=f"Request evaluator fallback applied: {reason}",
            gate=GateType.REQUEST,
            evaluator="fallback",
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"fallback_action": action.value},
        )


def _request_low_confidence_verdict(action: LowConfidenceAction) -> Verdict:
    if action == LowConfidenceAction.ALLOW:
        return Verdict.ALLOW
    if action == LowConfidenceAction.BLOCK:
        return Verdict.BLOCK
    return Verdict.CLARIFY


def _request_fallback_verdict(action: FallbackAction) -> Verdict:
    if action == FallbackAction.ALLOW:
        return Verdict.ALLOW
    if action == FallbackAction.CLARIFY:
        return Verdict.CLARIFY
    return Verdict.BLOCK
