from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aegisvault import AegisVault
from aegisvault.types import GateType, TerminatedBy, Verdict
from evaluation.scripts.eval_lib import (
    DatasetError,
    JsonlWriter,
    RequestCase,
    ResponseCase,
    build_metrics,
    completed_keys,
    ensure_output_dir,
    generate_run_id,
    is_quota_or_rate_limit_error,
    load_domain_assets,
    load_jsonl,
    no_secret_payload,
    overhead_ms,
    overhead_percent,
    progress_totals,
    read_jsonl,
    safe_latency_summary,
)
from evaluation.scripts.run_evaluation import _record_match
from evaluation.scripts.run_evaluation import _record_failure
from conftest import FakeEvaluator, MemoryAuditSink, decision


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _policy_dict(tmp_path: Path) -> dict:
    return {
        "version": "1.0",
        "application": {"name": "demo", "description": "Demo"},
        "purpose": "Help with demo support.",
        "allowed_topics": ["support"],
        "blocked_topics": ["code"],
        "gates": {
            "request": {"enabled": True, "allow_threshold": 0.8, "block_threshold": 0.8, "low_confidence_action": "clarify"},
            "response": {"enabled": True, "allow_threshold": 0.8, "block_threshold": 0.8, "low_confidence_action": "block"},
        },
        "evaluator": {"provider": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434", "timeout_seconds": 1, "temperature": 0},
        "audit": {"enabled": False, "output_path": str(tmp_path / "audit.jsonl"), "include_request_text": True, "include_response_text": True},
    }


def test_dataset_loading(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    _write_jsonl(path, [{"id": "r1", "domain": "demo", "type": "request", "category": "in_domain", "text": "help", "expected_request_verdict": "ALLOW"}])

    rows = load_jsonl(path, RequestCase)

    assert len(rows) == 1
    assert rows[0].id == "r1"


def test_invalid_dataset_row_handling(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "missing fields"}\n', encoding="utf-8")

    with pytest.raises(DatasetError):
        load_jsonl(path, RequestCase)


def test_policy_to_domain_mapping(tmp_path: Path) -> None:
    import yaml

    policy_dir = tmp_path / "policies"
    dataset_dir = tmp_path / "datasets"
    policy_dir.mkdir()
    (policy_dir / "demo.yaml").write_text(yaml.safe_dump(_policy_dict(tmp_path)), encoding="utf-8")
    _write_jsonl(dataset_dir / "demo" / "requests.jsonl", [{"id": "r1", "domain": "demo", "type": "request", "category": "in_domain", "text": "help", "expected_request_verdict": "ALLOW"}])
    _write_jsonl(dataset_dir / "demo" / "responses.jsonl", [{"id": "s1", "domain": "demo", "type": "response", "category": "in_domain", "source_prompt": "help", "text": "ok", "expected_response_verdict": "ALLOW"}])

    assets = load_domain_assets(["demo"], policy_dir, dataset_dir)

    assert assets["demo"].policy.application.name == "demo"


def test_unsafe_domain_name_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Unsafe domain name"):
        load_domain_assets(["../bad"], tmp_path / "policies", tmp_path / "datasets")


def test_request_metric_calculations_and_false_rates() -> None:
    metrics = build_metrics(
        [
            {"mode": "request", "domain": "d", "category": "in_domain", "match": True, "actual_verdict": "ALLOW", "decision_source": "ollama:x", "confidence": 0.9, "latency_ms": 10},
            {"mode": "request", "domain": "d", "category": "in_domain", "match": False, "actual_verdict": "BLOCK", "decision_source": "ollama:x", "confidence": 0.9, "latency_ms": 10},
            {"mode": "request", "domain": "d", "category": "out_of_domain", "match": False, "actual_verdict": "ALLOW", "decision_source": "ollama:x", "confidence": 0.9, "latency_ms": 10},
            {"mode": "request", "domain": "d", "category": "ambiguous", "match": True, "actual_verdict": "CLARIFY", "decision_source": "deterministic", "confidence": None, "latency_ms": 0},
        ]
    )
    request = metrics["overall"]["request"]

    assert request["matched_count"] == 2
    assert request["mismatched_count"] == 2
    assert request["false_positive_count"] == 1
    assert request["false_negative_count"] == 1
    assert request["ambiguous_case_count"] == 1
    assert request["deterministic_decision_count"] == 1


def test_response_metric_calculations() -> None:
    metrics = build_metrics(
        [
            {"mode": "response", "domain": "d", "category": "in_domain", "match": True, "actual_verdict": "ALLOW", "confidence": 0.9, "latency_ms": 10},
            {"mode": "response", "domain": "d", "category": "out_of_domain", "match": True, "actual_verdict": "BLOCK", "confidence": 0.9, "latency_ms": 10},
            {"mode": "response", "domain": "d", "category": "drifting", "match": False, "actual_verdict": "ALLOW", "confidence": 0.9, "latency_ms": 10},
        ]
    )

    response = metrics["overall"]["response"]
    assert response["response_false_negative_count"] == 1
    assert response["valid_response_acceptance_rate"] == 1.0


def test_overhead_calculations_and_zero_safety() -> None:
    assert overhead_ms(150, 100) == 50
    assert overhead_ms(80, 100) == -20
    assert overhead_percent(50, 100) == 50
    assert overhead_percent(50, 0) is None


def test_p95_latency_calculation() -> None:
    summary = safe_latency_summary([1, 2, 3, 4, 100])

    assert summary["p95_ms"] == 100


def test_progress_total_calculation(tmp_path: Path) -> None:
    class Asset:
        requests = [1, 2, 3]
        responses = [1, 2]

    totals = progress_totals({"d": Asset()}, ["request", "response", "end-to-end"], limit=2, runs=3)

    assert totals["request_cases"] == 2
    assert totals["response_cases"] == 2
    assert totals["total_executions"] == 18


def test_matched_counter_terms() -> None:
    counters = {"matched": 0, "mismatched": 0, "latency_total_ms": 0.0, "latency_count": 0}

    _record_match(counters, True, 10)
    _record_match(counters, False, 20)

    assert counters["matched"] == 1
    assert counters["mismatched"] == 1
    assert counters["latency_count"] == 2


def test_run_id_generation() -> None:
    run_id = generate_run_id()

    assert "T" in run_id
    assert "_" in run_id


def test_output_directory_creation(tmp_path: Path) -> None:
    result_dir = ensure_output_dir(tmp_path, "run1")

    assert result_dir.exists()


def test_jsonl_result_writing(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    JsonlWriter(path).write({"a": 1})

    assert read_jsonl(path) == [{"a": 1}]


def test_resume_behaviour(tmp_path: Path) -> None:
    path = tmp_path / "case_results.jsonl"
    JsonlWriter(path).write({"mode": "request", "id": "r1", "run_index": 0})

    assert ("request", "r1", 0) in completed_keys(tmp_path)


def test_gemini_not_called_when_request_gate_blocks(policy) -> None:
    called = 0

    def app(prompt: str) -> str:
        nonlocal called
        called += 1
        return "ok"

    guard = AegisVault(
        policy=policy,
        evaluator=FakeEvaluator([decision(Verdict.BLOCK, 0.9, GateType.REQUEST)]),
        audit_sink=MemoryAuditSink(),
    )

    result = guard.wrap(app)("Write code")

    assert called == 0
    assert result.terminated_by == TerminatedBy.REQUEST_GATE


def test_gemini_called_when_request_gate_allows(policy) -> None:
    called = 0

    def app(prompt: str) -> str:
        nonlocal called
        called += 1
        return "ok"

    guard = AegisVault(
        policy=policy,
        evaluator=FakeEvaluator([decision(Verdict.ALLOW, 0.9, GateType.REQUEST), decision(Verdict.ALLOW, 0.9, GateType.RESPONSE)]),
        audit_sink=MemoryAuditSink(),
    )

    result = guard.wrap(app)("Order help")

    assert called == 1
    assert result.terminated_by == TerminatedBy.APPLICATION


def test_response_gate_result_is_saved_shape() -> None:
    row = ResponseCase(id="s1", domain="d", type="response", category="in_domain", source_prompt="p", text="ok", expected_response_verdict="ALLOW")

    assert row.expected_response_verdict == Verdict.ALLOW


def test_api_failure_recording_metric() -> None:
    metrics = build_metrics([{"mode": "end-to-end", "domain": "d", "category": "in_domain", "error_type": "gemini", "pass": False}])

    assert metrics["overall"]["end_to_end"]["api_error_count"] == 1


def test_secrets_are_not_written_to_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")

    assert no_secret_payload({"model": "x"})
    assert not no_secret_payload({"leak": "super-secret"})


def test_quota_error_detection() -> None:
    assert is_quota_or_rate_limit_error("429 RESOURCE_EXHAUSTED quota exceeded")
    assert not is_quota_or_rate_limit_error("plain validation error")


def test_end_to_end_quota_failure_is_skipped(tmp_path: Path) -> None:
    case = RequestCase(
        id="r1",
        domain="demo",
        type="request",
        category="in_domain",
        text="hello",
        expected_request_verdict="ALLOW",
    )
    writers = {
        "failure": JsonlWriter(tmp_path / "failures.jsonl"),
        "case": JsonlWriter(tmp_path / "case_results.jsonl"),
    }
    counters = {"matched": 0, "mismatched": 0, "skipped": 0}

    _record_failure(
        writers,
        counters,
        case,
        "end-to-end",
        0,
        RuntimeError("Gemini request failed: 429 RESOURCE_EXHAUSTED quota exceeded"),
        False,
    )

    rows = read_jsonl(tmp_path / "case_results.jsonl")
    assert counters["skipped"] == 1
    assert counters["mismatched"] == 0
    assert rows[0]["skipped"] is True
    assert rows[0]["mismatched"] is False
