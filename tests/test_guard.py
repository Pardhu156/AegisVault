from __future__ import annotations

import pytest

from aegisvault import AegisVault
from aegisvault.exceptions import UnsupportedCallableError
from aegisvault.policy.models import DomainPolicy
from aegisvault.types import GateType, TerminatedBy, Verdict

from conftest import FakeEvaluator, MemoryAuditSink, decision


def test_protected_callable_not_called_for_blocked_requests(policy) -> None:
    called = 0

    def app(prompt: str) -> str:
        nonlocal called
        called += 1
        return prompt

    guard = AegisVault(
        policy=policy,
        evaluator=FakeEvaluator([decision(Verdict.BLOCK, 0.9, GateType.REQUEST)]),
        audit_sink=MemoryAuditSink(),
    )

    result = guard.wrap(app)("Write code")

    assert called == 0
    assert result.application_called is False
    assert result.terminated_by == TerminatedBy.REQUEST_GATE


def test_protected_callable_called_once_for_allowed_requests(policy) -> None:
    called = 0

    def app(prompt: str) -> str:
        nonlocal called
        called += 1
        return "Order status"

    evaluator = FakeEvaluator(
        [
            decision(Verdict.ALLOW, 0.9, GateType.REQUEST),
            decision(Verdict.ALLOW, 0.9, GateType.RESPONSE),
        ]
    )
    result = AegisVault(policy=policy, evaluator=evaluator, audit_sink=MemoryAuditSink()).wrap(app)("Where is my order?")

    assert called == 1
    assert result.final_response == "Order status"
    assert result.terminated_by == TerminatedBy.APPLICATION


def test_response_gate_blocks_final_response(policy) -> None:
    def app(prompt: str) -> str:
        return "Here is code."

    evaluator = FakeEvaluator(
        [
            decision(Verdict.ALLOW, 0.9, GateType.REQUEST),
            decision(Verdict.BLOCK, 0.9, GateType.RESPONSE),
        ]
    )

    result = AegisVault(policy=policy, evaluator=evaluator, audit_sink=MemoryAuditSink()).wrap(app)("Order help")

    assert result.application_called is True
    assert result.response_accepted is False
    assert result.was_modified is True
    assert result.terminated_by == TerminatedBy.RESPONSE_GATE


def test_callable_returning_non_string(policy) -> None:
    def app(prompt: str) -> int:
        return 123

    evaluator = FakeEvaluator(
        [
            decision(Verdict.ALLOW, 0.9, GateType.REQUEST),
            decision(Verdict.ALLOW, 0.9, GateType.RESPONSE),
        ]
    )

    result = AegisVault(policy=policy, evaluator=evaluator, audit_sink=MemoryAuditSink()).wrap(app)("Order help")

    assert result.final_response == "123"
    assert result.original_response == "123"


def test_disabled_request_gate(policy_dict) -> None:
    policy_dict["gates"]["request"]["enabled"] = False
    policy = DomainPolicy.model_validate(policy_dict)
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.9, GateType.RESPONSE)])

    result = AegisVault(policy=policy, evaluator=evaluator, audit_sink=MemoryAuditSink()).wrap(lambda p: "ok")("FORBIDDEN PHRASE")

    assert result.application_called is True
    assert result.request_decision.verdict == Verdict.ALLOW


def test_disabled_response_gate(policy_dict) -> None:
    policy_dict["gates"]["response"]["enabled"] = False
    policy = DomainPolicy.model_validate(policy_dict)
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.9, GateType.REQUEST)])

    result = AegisVault(policy=policy, evaluator=evaluator, audit_sink=MemoryAuditSink()).wrap(lambda p: "")("Order help")

    assert result.response_decision.verdict == Verdict.ALLOW
    assert result.final_response == ""


def test_async_callable_rejected(policy) -> None:
    async def app(prompt: str) -> str:
        return prompt

    with pytest.raises(UnsupportedCallableError):
        AegisVault(policy=policy, evaluator=FakeEvaluator(), audit_sink=MemoryAuditSink()).wrap(app)
