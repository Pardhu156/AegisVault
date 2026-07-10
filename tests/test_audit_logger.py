from __future__ import annotations

import json
from pathlib import Path

from aegisvault import AegisVault
from aegisvault.audit import JsonLineAuditSink
from aegisvault.types import GateType, TerminatedBy, Verdict

from conftest import FakeEvaluator, MemoryAuditSink, decision


def test_audit_log_creation(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "audit.jsonl"
    sink = JsonLineAuditSink(path)
    sink.record({"hello": "world"})

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"hello": "world"}


def test_audit_event_contents(policy) -> None:
    sink = MemoryAuditSink()
    evaluator = FakeEvaluator(
        [
            decision(Verdict.ALLOW, 0.9, GateType.REQUEST),
            decision(Verdict.ALLOW, 0.9, GateType.RESPONSE),
        ]
    )

    result = AegisVault(policy=policy, evaluator=evaluator, audit_sink=sink).wrap(lambda p: "ok")("Where is my order?")

    assert result.terminated_by == TerminatedBy.APPLICATION
    assert len(sink.events) == 2
    assert sink.events[0]["gate"] == "request"
    assert sink.events[0]["input_text"] == "Where is my order?"
    assert sink.events[0]["terminated_by"] == "APPLICATION"
    assert sink.events[1]["gate"] == "response"
    assert sink.events[1]["generated_response"] == "ok"
    assert sink.events[1]["terminated_by"] == "APPLICATION"
