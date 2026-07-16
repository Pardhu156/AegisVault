from __future__ import annotations

import json
from collections import Counter

import pytest

from aegisvault.runtime.goal_vault import GoalEmbeddingError
from evaluation.agentdojo import run_pilot_benchmark as harness


def test_agentdojo_suite_specific_tool_metadata_marks_send_as_risky() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "send_slack_message")

    assert metadata.risk_level == "medium"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True
    assert "strict_verification" in metadata.required_permissions


def test_agentdojo_suite_specific_tool_metadata_marks_search_as_low_risk() -> None:
    metadata = harness._agentdojo_tool_metadata("workspace", object(), "search_workspace")

    assert metadata.risk_level == "low"
    assert metadata.side_effect_level.value == "read"
    assert metadata.requires_approval is False


def test_agentdojo_banking_read_tools_are_not_treated_as_financial_writes() -> None:
    metadata = harness._agentdojo_tool_metadata("banking", object(), "get_most_recent_transactions")

    assert metadata.risk_level == "low"
    assert metadata.side_effect_level.value == "read"
    assert metadata.requires_approval is False


def test_agentdojo_banking_money_movement_remains_strict() -> None:
    metadata = harness._agentdojo_tool_metadata("banking", object(), "send_money")

    assert metadata.risk_level == "high"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True
    assert "strict_verification" in metadata.required_permissions


def test_agentdojo_slack_read_messages_are_low_risk_reads() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "read_channel_messages")

    assert metadata.risk_level == "low"
    assert metadata.side_effect_level.value == "read"
    assert metadata.requires_approval is False


def test_agentdojo_slack_send_messages_remain_strict() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "send_channel_message")

    assert metadata.risk_level == "medium"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True
    assert "strict_verification" in metadata.required_permissions


def test_duplicate_result_rows_are_compacted(tmp_path) -> None:
    path = tmp_path / "protected_results.jsonl"
    rows = {
        "case-a": {"case_id": "case-a", "value": 2},
        "case-b": {"case_id": "case-b", "value": 3},
    }
    path.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "case-a", "value": 1}),
                json.dumps({"case_id": "case-a", "value": 2}),
                json.dumps({"case_id": "case-b", "value": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    harness._rewrite_unique_jsonl(path, rows)

    compacted = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert compacted == [{"case_id": "case-a", "value": 2}, {"case_id": "case-b", "value": 3}]


def test_scaled_sampler_selects_clean_and_attack_cases_without_duplicates() -> None:
    cases = harness.select_cases(
        limit=None,
        suites=("workspace", "slack", "banking", "travel"),
        attack="direct",
        clean_limit=20,
        attack_limit=100,
        balanced_by_suite=True,
        seed=7,
        benchmark_version="v1.2.2",
    )

    assert len(cases) == 120
    assert sum(1 for case in cases if case.case_type == "benign") == 20
    assert sum(1 for case in cases if case.case_type == "attack") == 100
    assert len({case.case_id for case in cases}) == 120
    clean_by_suite = Counter(case.suite for case in cases if case.case_type == "benign")
    attack_by_suite = Counter(case.suite for case in cases if case.case_type == "attack")
    assert clean_by_suite == {"workspace": 5, "slack": 5, "banking": 5, "travel": 5}
    assert attack_by_suite == {"workspace": 25, "slack": 25, "banking": 25, "travel": 25}


def test_scaled_sampler_can_interleave_clean_and_attack_cases() -> None:
    cases = harness.select_cases(
        limit=None,
        suites=("workspace", "slack", "banking", "travel"),
        attack="direct",
        clean_limit=4,
        attack_limit=4,
        balanced_by_suite=True,
        case_layout="interleave-types",
        seed=7,
        benchmark_version="v1.2.2",
    )

    assert len(cases) == 8
    assert [case.case_type for case in cases[:4]] == ["benign", "attack", "benign", "attack"]


def test_case_record_marks_legit_cases_with_null_injection() -> None:
    case = harness.select_cases(
        limit=1,
        suites=("workspace",),
        attack="direct",
        clean_limit=1,
        attack_limit=0,
        balanced_by_suite=True,
        seed=7,
        benchmark_version="v1.2.2",
    )[0]

    record = harness._case_record(case)
    assert record["case_type"] == "benign"
    assert record["injection_task_id"] is None


def test_missing_real_embedder_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEmbedder:
        model_name = "all-MiniLM-L6-v2"
        dimension = 384

        def embed(self, text: str):
            raise GoalEmbeddingError("model unavailable")

    monkeypatch.setattr(harness, "_production_embedder", lambda: BrokenEmbedder())

    with pytest.raises(SystemExit, match="Production embedder unavailable"):
        harness._verify_production_embedder()


def test_action_rows_include_flattened_decision_fields(tmp_path) -> None:
    path = tmp_path / "actions.jsonl"
    row = {
        "phase": "protected",
        "case_id": "case-a",
        "suite": "workspace",
        "case_type": "benign",
        "user_task_id": "user_task_1",
        "injection_task_id": None,
        "utility": True,
        "injection_success": False,
        "middleware": {
            "action_traces": [
                {
                    "tool_name": "search_files",
                    "risk_classification": {
                        "risk_level": "low",
                        "side_effect_level": "read",
                        "requires_approval": False,
                    },
                    "sentinel": {"decision": "allow"},
                    "action_gate": {
                        "verdict": "EXECUTE",
                        "decision_source": "COSINE",
                        "ollama_called": False,
                        "goal_similarity": 0.91,
                    },
                    "final_result": "EXECUTE",
                    "executed": True,
                }
            ]
        },
    }

    harness._append_action_rows(path, row)

    written = json.loads(path.read_text(encoding="utf-8").strip())
    assert written["risk_level"] == "low"
    assert written["side_effect_level"] == "read"
    assert written["sentinel_verdict"] == "allow"
    assert written["action_gate_verdict"] == "EXECUTE"
    assert written["ollama_called"] is False
