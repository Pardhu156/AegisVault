"""Public Layer 0 validator."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Callable, Mapping
from uuid import uuid4

from aegisvault.audit import AuditSink, NullAuditSink
from aegisvault.layer0.models import (
    Layer0Action,
    Layer0Checkpoint,
    Layer0Decision,
    Layer0RequestInput,
    Layer0RiskLevel,
    Layer0RuleResult,
    Layer0ToolCallInput,
)
from aegisvault.layer0.rules import request_rules, tool_rules
from aegisvault.policy.models import DomainPolicy, Layer0FailMode


class Layer0Validator:
    """Deterministic structural validator for requests and tool calls."""

    def __init__(
        self,
        *,
        policy: DomainPolicy,
        audit_sink: AuditSink | None = None,
        trusted_goal_exists: Callable[[str | None], bool] | None = None,
        tool_catalog: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.config = policy.layer0
        self.audit_sink = audit_sink or NullAuditSink()
        self.trusted_goal_exists = trusted_goal_exists or (lambda session_id: False)
        self.tool_catalog = dict(tool_catalog or {})

    @property
    def enabled(self) -> bool:
        """Whether Layer 0 is enabled by policy."""

        return self.config.enabled

    def validate_request(
        self,
        *,
        session_id: str | None,
        request_text: Any,
        domain: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        requested_goal_update: Any = None,
    ) -> Layer0Decision:
        """Validate an incoming request before Request Gate evaluation."""

        started = time.perf_counter()
        context = Layer0RequestInput(
            session_id=session_id,
            request_text=request_text,
            domain=domain,
            policy_name=self.policy.application.name,
            metadata=metadata or {},
            requested_goal_update=requested_goal_update,
            trusted_goal_exists=self.trusted_goal_exists(session_id),
        )
        try:
            decision = _aggregate(
                checkpoint=Layer0Checkpoint.REQUEST,
                results=request_rules(context, self.config),
                stop_on_first_block=self.config.stop_on_first_block,
            )
        except Exception as exc:
            decision = _unexpected_decision(Layer0Checkpoint.REQUEST, self.config.fail_mode, exc)
        self._audit(decision, started=started, session_id=session_id, metadata=metadata or {})
        return decision

    def validate_tool_call(
        self,
        *,
        session_id: str | None,
        tool_name: str | None,
        arguments: Any,
        domain: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tool_catalog: Mapping[str, Any] | None = None,
    ) -> Layer0Decision:
        """Validate a proposed tool call before Action Gate evaluation."""

        started = time.perf_counter()
        context = Layer0ToolCallInput(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            domain=domain,
            policy_name=self.policy.application.name,
            metadata=metadata or {},
            tool_catalog=tool_catalog if tool_catalog is not None else self.tool_catalog,
        )
        try:
            decision = _aggregate(
                checkpoint=Layer0Checkpoint.TOOL_CALL,
                results=tool_rules(context, self.config),
                stop_on_first_block=self.config.stop_on_first_block,
            )
        except Exception as exc:
            decision = _unexpected_decision(Layer0Checkpoint.TOOL_CALL, self.config.fail_mode, exc)
        self._audit(decision, started=started, session_id=session_id, tool_name=tool_name, arguments=arguments)
        return decision

    def _audit(
        self,
        decision: Layer0Decision,
        *,
        started: float,
        session_id: str | None,
        metadata: Mapping[str, Any] | None = None,
        tool_name: str | None = None,
        arguments: Any = None,
    ) -> None:
        event = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "event": f"layer0.{decision.checkpoint.value}",
            "session_id": session_id,
            "checkpoint": decision.checkpoint.value,
            "decision": decision.decision.value,
            "risk_level": decision.risk_level.value,
            "matched_rule_ids": [rule.rule_id for rule in decision.matched_rules],
            "reason": decision.reason,
            "policy": self.policy.application.name,
            "tool_name": tool_name,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "redacted_fields": _collect_redacted_fields(decision),
            "metadata": redact_sensitive(metadata or {}, self.config.tools.sensitive_argument_keys),
        }
        if arguments is not None:
            event["arguments"] = redact_sensitive(arguments, self.config.tools.sensitive_argument_keys)
        try:
            self.audit_sink.record(event)
        except Exception:
            return None


def redact_sensitive(value: Any, sensitive_keys: list[str] | tuple[str, ...]) -> Any:
    """Recursively redact configured sensitive keys."""

    lowered = {key.lower() for key in sensitive_keys}
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_str = str(key)
            if key_str.lower() in lowered:
                redacted[key_str] = "[redacted]"
            else:
                redacted[key_str] = redact_sensitive(child, sensitive_keys)
        return redacted
    if isinstance(value, list | tuple):
        return [redact_sensitive(item, sensitive_keys) for item in value]
    return value


def _aggregate(
    *,
    checkpoint: Layer0Checkpoint,
    results: list[Layer0RuleResult],
    stop_on_first_block: bool,
) -> Layer0Decision:
    matched: list[Layer0RuleResult] = []
    for result in results:
        if not result.matched:
            continue
        matched.append(result)
        if stop_on_first_block and result.action == Layer0Action.BLOCK:
            break
    if not matched:
        return Layer0Decision(
            allowed=True,
            decision=Layer0Action.ALLOW,
            risk_level=Layer0RiskLevel.NONE,
            checkpoint=checkpoint,
            reason="Layer 0 validation passed.",
        )
    blocking = [result for result in matched if result.action == Layer0Action.BLOCK]
    if blocking:
        highest = _highest_risk(blocking)
        first = blocking[0]
        return Layer0Decision(
            allowed=False,
            decision=Layer0Action.BLOCK,
            risk_level=highest,
            checkpoint=checkpoint,
            rule_id=first.rule_id,
            reason=first.reason,
            matched_rules=tuple(matched),
        )
    highest = _highest_risk(matched)
    return Layer0Decision(
        allowed=True,
        decision=Layer0Action.WARN,
        risk_level=highest,
        checkpoint=checkpoint,
        rule_id=matched[0].rule_id,
        reason=matched[0].reason,
        matched_rules=tuple(matched),
    )


def _unexpected_decision(checkpoint: Layer0Checkpoint, fail_mode: Layer0FailMode, exc: Exception) -> Layer0Decision:
    if fail_mode == Layer0FailMode.OPEN:
        return Layer0Decision(
            allowed=True,
            decision=Layer0Action.WARN,
            risk_level=Layer0RiskLevel.MEDIUM,
            checkpoint=checkpoint,
            rule_id="L0_INTERNAL_ERROR",
            reason="Layer 0 encountered an internal error and failed open.",
            matched_rules=(
                Layer0RuleResult(
                    rule_id="L0_INTERNAL_ERROR",
                    matched=True,
                    action=Layer0Action.WARN,
                    risk_level=Layer0RiskLevel.MEDIUM,
                    reason="Layer 0 internal error.",
                    metadata={"error_type": exc.__class__.__name__},
                ),
            ),
        )
    return Layer0Decision(
        allowed=False,
        decision=Layer0Action.BLOCK,
        risk_level=Layer0RiskLevel.HIGH,
        checkpoint=checkpoint,
        rule_id="L0_INTERNAL_ERROR",
        reason="Layer 0 encountered an internal error and failed closed.",
        matched_rules=(
            Layer0RuleResult(
                rule_id="L0_INTERNAL_ERROR",
                matched=True,
                action=Layer0Action.BLOCK,
                risk_level=Layer0RiskLevel.HIGH,
                reason="Layer 0 internal error.",
                metadata={"error_type": exc.__class__.__name__},
            ),
        ),
    )


def _highest_risk(results: list[Layer0RuleResult]) -> Layer0RiskLevel:
    rank = {
        Layer0RiskLevel.NONE: 0,
        Layer0RiskLevel.LOW: 1,
        Layer0RiskLevel.MEDIUM: 2,
        Layer0RiskLevel.HIGH: 3,
        Layer0RiskLevel.CRITICAL: 4,
    }
    return max((result.risk_level for result in results), key=lambda item: rank[item], default=Layer0RiskLevel.NONE)


def _collect_redacted_fields(decision: Layer0Decision) -> list[str]:
    fields: list[str] = []
    for rule in decision.matched_rules:
        raw_fields = rule.metadata.get("redacted_fields") if isinstance(rule.metadata, Mapping) else None
        if isinstance(raw_fields, list | tuple):
            fields.extend(str(item) for item in raw_fields)
    return fields
