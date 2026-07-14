"""Public AegisVault guard API."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

from aegisvault.audit import AuditSink, JsonLineAuditSink, NullAuditSink
from aegisvault.evaluators import OllamaScopeEvaluator, ScopeEvaluator
from aegisvault.exceptions import UnsupportedCallableError
from aegisvault.gates import RequestGate, ResponseGate
from aegisvault.layer0 import Layer0Validator
from aegisvault.policy import DomainPolicy, load_policy
from aegisvault.types import EvaluationContext, GateDecision, GateType, GuardResult, TerminatedBy, Verdict

logger = logging.getLogger(__name__)


class AegisVault:
    """Domain-specific guardrail middleware for synchronous Python callables."""

    def __init__(
        self,
        *,
        policy: DomainPolicy,
        evaluator: ScopeEvaluator | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self.policy = policy
        self.evaluator = evaluator or self._build_evaluator(policy)
        self.request_gate = RequestGate(policy, self.evaluator)
        self.response_gate = ResponseGate(policy, self.evaluator)
        self.audit_sink = audit_sink if audit_sink is not None else self._build_audit_sink(policy)
        self.layer0_validator = Layer0Validator(policy=policy, audit_sink=self.audit_sink)

    @classmethod
    def from_policy(cls, path: str | Path) -> "AegisVault":
        """Load a YAML policy and construct an AegisVault guard."""

        return cls(policy=load_policy(path))

    def wrap(self, app: Callable[[str], str]) -> Callable[[str], GuardResult]:
        """Wrap a synchronous callable that accepts a string and returns a string."""

        if inspect.iscoroutinefunction(app):
            raise UnsupportedCallableError("AegisVault Stage 1 supports synchronous callables only")
        if not callable(app):
            raise UnsupportedCallableError("AegisVault can only wrap callable objects")

        @wraps(app)
        def guarded(prompt: str, *, session_id: str | None = None, metadata: dict[str, Any] | None = None) -> GuardResult:
            if not isinstance(prompt, str):
                prompt = str(prompt)

            layer0_decision = None
            if self.layer0_validator.enabled:
                layer0_decision = self.layer0_validator.validate_request(
                    session_id=session_id,
                    request_text=prompt,
                    domain=(metadata or {}).get("domain"),
                    metadata=metadata or {},
                    requested_goal_update=(metadata or {}).get("requested_goal_update"),
                )
                if not layer0_decision.allowed:
                    final_response = self.policy.messages.request_blocked
                    return GuardResult(
                        final_response=final_response,
                        request_decision=None,
                        response_decision=None,
                        application_called=False,
                        request_accepted=False,
                        response_accepted=None,
                        was_modified=True,
                        terminated_by=TerminatedBy.LAYER0,
                        metadata={**(metadata or {}), "layer0_request_decision": _layer0_summary(layer0_decision)},
                    )

            context = EvaluationContext(request_text=prompt, session_id=session_id, metadata=metadata or {})
            request_decision = self.request_gate.evaluate(prompt, context)
            if layer0_decision is not None:
                request_decision.metadata["layer0_request_decision"] = _layer0_summary(layer0_decision)
            if request_decision.verdict != Verdict.ALLOW:
                final_response = self._request_fallback_text(request_decision)
                result = GuardResult(
                    final_response=final_response,
                    request_decision=request_decision,
                    response_decision=None,
                    application_called=False,
                    request_accepted=False,
                    response_accepted=None,
                    was_modified=True,
                    terminated_by=TerminatedBy.REQUEST_GATE,
                    metadata=metadata or {},
                )
                self._audit(
                    request_decision,
                    input_text=prompt,
                    final_response=final_response,
                    session_id=session_id,
                    result=result,
                )
                return result

            raw_response = app(prompt)
            application_response = raw_response if isinstance(raw_response, str) else str(raw_response)
            response_context = EvaluationContext(
                request_text=prompt,
                response_text=application_response,
                session_id=session_id,
                metadata=metadata or {},
            )
            response_decision = self.response_gate.evaluate(application_response, response_context)

            final_response, response_accepted, was_modified, terminated_by = self._finalize_response(
                response_decision,
                application_response,
            )
            result = GuardResult(
                final_response=final_response,
                request_decision=request_decision,
                response_decision=response_decision,
                application_called=True,
                request_accepted=True,
                response_accepted=response_accepted,
                was_modified=was_modified,
                terminated_by=terminated_by,
                original_response=application_response,
                metadata={**(metadata or {}), **({"layer0_request_decision": _layer0_summary(layer0_decision)} if layer0_decision else {})},
            )
            self._audit(
                request_decision,
                input_text=prompt,
                application_called=True,
                session_id=session_id,
                result=result,
            )
            self._audit(
                response_decision,
                input_text=prompt,
                generated_response=application_response,
                final_response=final_response,
                application_called=True,
                session_id=session_id,
                result=result,
            )
            return result

        return guarded

    def wrap_text(self, app: Callable[[str], str]) -> Callable[[str], str]:
        """Wrap a callable and return only the final response text."""

        guarded = self.wrap(app)

        @wraps(app)
        def text_only(prompt: str, *, session_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
            return guarded(prompt, session_id=session_id, metadata=metadata).final_response

        return text_only

    def _build_evaluator(self, policy: DomainPolicy) -> ScopeEvaluator:
        if policy.evaluator.provider == "ollama":
            return OllamaScopeEvaluator.from_policy(policy)
        raise ValueError(f"Unsupported evaluator provider: {policy.evaluator.provider}")

    def _build_audit_sink(self, policy: DomainPolicy) -> AuditSink:
        if not policy.audit.enabled:
            return NullAuditSink()
        return JsonLineAuditSink(policy.audit.output_path)

    def _request_fallback_text(self, decision: GateDecision) -> str:
        if decision.verdict == Verdict.CLARIFY:
            return self.policy.messages.request_clarify
        return self.policy.messages.request_blocked

    def _finalize_response(self, decision: GateDecision, application_response: str) -> tuple[str, bool, bool, TerminatedBy]:
        if decision.verdict == Verdict.ALLOW:
            return application_response, True, False, TerminatedBy.APPLICATION
        if decision.verdict == Verdict.REPLACE:
            return self.policy.messages.response_replaced, False, True, TerminatedBy.RESPONSE_GATE
        return self.policy.messages.response_blocked, False, True, TerminatedBy.RESPONSE_GATE

    def _audit(
        self,
        decision: GateDecision,
        *,
        input_text: str,
        generated_response: str | None = None,
        final_response: str | None = None,
        application_called: bool = False,
        session_id: str | None = None,
        result: GuardResult | None = None,
        result_terminated_by: TerminatedBy | None = None,
    ) -> None:
        gate = decision.gate
        include_text = (
            self.policy.audit.include_request_text
            if gate == GateType.REQUEST
            else self.policy.audit.include_response_text
        )
        terminated_by = result.terminated_by if result is not None else result_terminated_by
        event: dict[str, Any] = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "application": self.policy.application.name,
            "gate": gate.value,
            "verdict": decision.verdict.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "latency_ms": decision.latency_ms,
            "evaluator": decision.evaluator,
            "application_called": application_called,
            "session_id": session_id,
            "terminated_by": terminated_by.value if terminated_by is not None else None,
            "metadata": decision.metadata,
        }
        if include_text:
            event["input_text"] = input_text
            if generated_response is not None:
                event["generated_response"] = generated_response
            if final_response is not None:
                event["final_response"] = final_response
        try:
            self.audit_sink.record(event)
        except Exception:
            logger.exception("Failed to write AegisVault audit event")


def _layer0_summary(decision: Any) -> dict[str, Any]:
    return {
        "decision": decision.decision.value,
        "allowed": decision.allowed,
        "risk_level": decision.risk_level.value,
        "checkpoint": decision.checkpoint.value,
        "rule_id": decision.rule_id,
        "matched_rule_ids": [rule.rule_id for rule in decision.matched_rules],
        "reason": decision.reason,
    }
