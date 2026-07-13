from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aegisvault.email_agent import EmailStore, build_email_tool_registry


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "email"


def test_email_dataset_loads_threaded_messages() -> None:
    store = EmailStore(DATASET, persist_sent=False)
    assert len(store.inbox) == 50
    assert len(store.sent) >= 8
    assert len(store.drafts) == 2
    assert len(store.contacts) >= 10
    alpha_thread = [message for message in store.messages if message.thread_id == "thread_alpha_priorities"]
    assert len(alpha_thread) >= 4
    assert any(message.from_email == "me@aegisvault.local" for message in alpha_thread)


def test_search_read_and_thread_summary() -> None:
    store = EmailStore(DATASET, persist_sent=False)
    results = store.search(query="Project Alpha", limit=5)
    assert results
    assert any("Project Alpha" in item["subject"] or "project_alpha" in item["labels"] for item in results)
    full = store.read_email("email_019")
    assert "second-half priorities" in full["body"]
    summary = store.summarize(thread_id="thread_boston_travel")
    assert summary["message_count"] == 4
    assert "Boston" in " ".join(summary["subjects"])


def test_email_tool_registry_executes_tools() -> None:
    registry = build_email_tool_registry(DATASET, persist_sent=False)
    names = {tool.name for tool in registry.list_tools()}
    assert {"search_email", "read_email", "summarize_email", "draft_email", "send_email", "delete_email", "archive_email", "list_unread", "list_threads"} <= names
    result = registry.execute("search_email", {"query": "Amazon", "limit": 3})
    assert result.error is None
    assert any(item["id"] == "email_011" for item in result.result)
    fuzzy = registry.execute("search_email", {"query": "invoice", "sender": "amazon.com", "limit": 3})
    assert fuzzy.error is None
    assert any(item["id"] == "email_011" for item in fuzzy.result)
    broad_domain = registry.execute("search_email", {"sender": "example.com", "limit": 20})
    assert broad_domain.error is None
    assert broad_domain.result == []


def test_send_email_is_simulated_without_real_delivery(tmp_path: Path) -> None:
    dataset = tmp_path / "email"
    _copy_dataset(DATASET, dataset)
    store = EmailStore(dataset, persist_sent=True)
    sent = store.send_email(to="manager", subject="Simulated check", body="This stays local.")
    assert sent["to"] == ["maya.rao@acme.example"]
    sent_file = dataset / "sent" / "sent_emails.jsonl"
    assert "Simulated check" in sent_file.read_text(encoding="utf-8")


def test_invalid_contact_and_missing_email_errors() -> None:
    registry = build_email_tool_registry(DATASET, persist_sent=False)
    missing = registry.execute("read_email", {"email_id": "missing"})
    assert missing.error is not None
    invalid = registry.execute("send_email", {"to": "not-a-contact", "subject": "Hi", "body": "Body"})
    assert invalid.error is not None


def test_cli_lists_email_tools() -> None:
    completed = subprocess.run(
        [sys.executable, "run_agent.py", "--email-agent", "--list-tools"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "search_email" in completed.stdout
    assert "list_threads" in completed.stdout


def test_email_evaluation_mock_generates_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            "evaluation/agent_runtime/scripts/run_email_agent_evaluation.py",
            "--mock",
            "--limit",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Report folder:" in completed.stdout
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["total_tasks"] == 3
    assert metrics["tool_execution_count"] >= 3
    assert (run_dirs[0] / "conversation_traces.jsonl").exists()


def _copy_dataset(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
