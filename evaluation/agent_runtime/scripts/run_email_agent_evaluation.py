from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvault.agent_runtime import AgentRuntime, JsonlTraceLogger, OllamaChatClient
from aegisvault.agent_runtime.ollama_client import OllamaChatResult
from aegisvault.email_agent import EMAIL_AGENT_SYSTEM_PROMPT, build_email_tool_registry


class KeywordMockEmailClient:
    model = "mock-email-agent"

    def __init__(self) -> None:
        self._pending_response = "Done."

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> OllamaChatResult:
        last = messages[-1]
        if last.get("role") == "tool":
            return OllamaChatResult(payload={"message": {"role": "assistant", "content": self._pending_response}}, latency_ms=0.2)
        prompt = str(last.get("content", "")).lower()
        calls: list[dict[str, Any]]
        if "unread" in prompt and "summarize" in prompt:
            calls = [
                {"function": {"name": "list_unread", "arguments": {"limit": 5}}},
                {"function": {"name": "summarize_email", "arguments": {}}},
            ]
            self._pending_response = "Summarized unread emails from the synthetic mailbox."
        elif "unread" in prompt:
            calls = [{"function": {"name": "list_unread", "arguments": {"limit": 5}}}]
            self._pending_response = "Listed unread emails from the synthetic mailbox."
        elif "thread" in prompt and "summarize" in prompt:
            calls = [
                {"function": {"name": "list_threads", "arguments": {"query": _query(prompt), "limit": 5}}},
                {"function": {"name": "summarize_email", "arguments": {"query": _query(prompt)}}},
            ]
            self._pending_response = "Summarized matching conversation threads."
        elif "thread" in prompt:
            calls = [{"function": {"name": "list_threads", "arguments": {"query": _query(prompt), "limit": 5}}}]
            self._pending_response = "Listed matching conversation threads."
        elif "summarize" in prompt:
            calls = [
                {"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 5}}},
                {"function": {"name": "summarize_email", "arguments": {"query": _query(prompt)}}},
            ]
            self._pending_response = "Summarized matching conversation threads."
        elif "status" in prompt and "draft" not in prompt and "reply" not in prompt:
            calls = [
                {"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 5}}},
                {"function": {"name": "summarize_email", "arguments": {"query": _query(prompt)}}},
            ]
            self._pending_response = "Searched and summarized the current status."
        elif "draft" in prompt or "reply" in prompt:
            calls = []
            if "reply" in prompt or "onboarding" in prompt:
                calls.append({"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 3}}})
            calls.append({"function": {"name": "draft_email", "arguments": {"to": _recipient(prompt), "subject": "Draft response", "body": "Thanks, I will follow up on this."}}})
            self._pending_response = "Created a simulated draft email."
        elif "send" in prompt:
            calls = [
                {"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 3}}},
                {"function": {"name": "send_email", "arguments": {"to": "finance", "subject": "Question about invoice", "body": "Can you confirm the invoice status?"}}},
            ]
            self._pending_response = "Sent a simulated email in the local sent dataset."
        elif "archive" in prompt:
            calls = [
                {"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 3}}},
                {"function": {"name": "archive_email", "arguments": {"email_id": "email_025"}}},
            ]
            self._pending_response = "Archived the selected synthetic email."
        elif "delete" in prompt:
            calls = [
                {"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 3}}},
                {"function": {"name": "delete_email", "arguments": {"email_id": "email_026"}}},
            ]
            self._pending_response = "Deleted the selected synthetic email in memory."
        elif _has_word(prompt, "read"):
            calls = [
                {"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 3}}},
                {"function": {"name": "read_email", "arguments": {"email_id": _read_id(prompt)}}},
            ]
            self._pending_response = "Read the requested email."
        else:
            calls = [{"function": {"name": "search_email", "arguments": {"query": _query(prompt), "limit": 5}}}]
            self._pending_response = "Found matching synthetic emails."
        return OllamaChatResult(payload={"message": {"role": "assistant", "content": "", "tool_calls": calls}}, latency_ms=0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 4.1 Email Agent.")
    parser.add_argument("--dataset", default="datasets/email")
    parser.add_argument("--tasks", default="datasets/email/tasks/email_tasks.jsonl")
    parser.add_argument("--output-dir", default="reports/email_agent")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock model for fast local tests.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "conversation_traces.jsonl"
    runtime = AgentRuntime(
        client=KeywordMockEmailClient()
        if args.mock
        else OllamaChatClient(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            num_predict=args.num_predict,
        ),
        tools=build_email_tool_registry(args.dataset, persist_sent=not args.mock),
        trace_logger=JsonlTraceLogger(trace_path),
        system_prompt=EMAIL_AGENT_SYSTEM_PROMPT,
        max_tool_rounds=5,
    )

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with (run_dir / "task_results.jsonl").open("w", encoding="utf-8") as results_file:
        with tqdm(total=len(tasks), desc="email-agent evaluation", dynamic_ncols=True) as progress:
            for task in tasks:
                case_started = time.perf_counter()
                progress.set_postfix_str(f"current={task['id']}")
                try:
                    result = runtime.run(task["task"])
                    tool_names = [record.tool_name for record in result.tool_records]
                    errors = [record.error for record in result.tool_records if record.error]
                    expected = task.get("expected_tools", [])
                    tool_match = all(tool in tool_names for tool in expected)
                    content_match = _content_matches(result.final_response, task.get("success_keywords", []))
                    success = not errors and bool(result.final_response.strip()) and (tool_match or content_match)
                    record = {
                        "id": task["id"],
                        "task": task["task"],
                        "expected_tools": expected,
                        "actual_tools": tool_names,
                        "tool_selection_match": tool_match,
                        "content_match": content_match,
                        "success": success,
                        "final_response": result.final_response,
                        "latency_ms": (time.perf_counter() - case_started) * 1000,
                        "tool_errors": errors,
                        "trace_id": result.trace.trace_id,
                    }
                except Exception as exc:
                    record = {
                        "id": task.get("id"),
                        "task": task.get("task"),
                        "expected_tools": task.get("expected_tools", []),
                        "actual_tools": [],
                        "tool_selection_match": False,
                        "content_match": False,
                        "success": False,
                        "final_response": "",
                        "latency_ms": (time.perf_counter() - case_started) * 1000,
                        "tool_errors": [f"{exc.__class__.__name__}: {exc}"],
                        "trace_id": None,
                    }
                results.append(record)
                results_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                results_file.flush()
                if args.verbose:
                    tqdm.write(f"{record['id']}: success={record['success']} tools={record['actual_tools']}")
                avg_latency = statistics.fmean(float(item["latency_ms"]) for item in results)
                progress.set_postfix(
                    current=record["id"],
                    passed=sum(1 for item in results if item["success"]),
                    failed=sum(1 for item in results if not item["success"]),
                    avg=f"{avg_latency:.0f}ms",
                )
                progress.update(1)

    metrics = _metrics(results, total_runtime_ms=(time.perf_counter() - started) * 1000, model=args.model, mock=args.mock)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "runtime_summary.md").write_text(_summary(run_id, args, metrics, run_dir), encoding="utf-8")
    print(f"Report folder: {run_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["runtime_failures"] == 0 else 1


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            for field in ("id", "task", "expected_tools"):
                if field not in payload:
                    raise ValueError(f"missing {field!r} in {path}:{line_no}")
            tasks.append(payload)
    return tasks


def _metrics(results: list[dict[str, Any]], *, total_runtime_ms: float, model: str, mock: bool) -> dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in results]
    tool_counter: Counter[str] = Counter(tool for item in results for tool in item["actual_tools"])
    successes = sum(1 for item in results if item["success"])
    tool_matches = sum(1 for item in results if item["tool_selection_match"])
    failures = sum(1 for item in results if item["tool_errors"])
    content_matches = sum(1 for item in results if item.get("content_match"))
    return {
        "model": "mock-email-agent" if mock else model,
        "total_tasks": len(results),
        "task_success_rate": successes / len(results) if results else 0,
        "tool_selection_accuracy": tool_matches / len(results) if results else 0,
        "content_match_rate": content_matches / len(results) if results else 0,
        "average_latency_ms": statistics.fmean(latencies) if latencies else 0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "min_latency_ms": min(latencies) if latencies else 0,
        "tool_execution_count": sum(tool_counter.values()),
        "average_tools_per_task": sum(tool_counter.values()) / len(results) if results else 0,
        "per_tool_usage": dict(sorted(tool_counter.items())),
        "runtime_failures": failures,
        "total_runtime_ms": total_runtime_ms,
    }


def _summary(run_id: str, args: argparse.Namespace, metrics: dict[str, Any], run_dir: Path) -> str:
    return "\n".join(
        [
            f"# Stage 4.1 Email Agent Evaluation: {run_id}",
            "",
            f"- Dataset: `{args.dataset}`",
            f"- Tasks: `{args.tasks}`",
            f"- Model: `{metrics['model']}`",
            f"- Total tasks: {metrics['total_tasks']}",
            f"- Task success rate: {metrics['task_success_rate']:.2%}",
            f"- Tool selection accuracy: {metrics['tool_selection_accuracy']:.2%}",
            f"- Content match rate: {metrics['content_match_rate']:.2%}",
            f"- Average latency: {metrics['average_latency_ms']:.1f} ms",
            f"- Average tools per task: {metrics['average_tools_per_task']:.2f}",
            f"- Runtime failures: {metrics['runtime_failures']}",
            "",
            "## Per-tool Usage",
            "",
            *[f"- {name}: {count}" for name, count in metrics["per_tool_usage"].items()],
            "",
            f"Output folder: `{run_dir}`",
        ]
    )


def _recipient(prompt: str) -> str:
    if "manager" in prompt or "maya" in prompt:
        return "manager"
    if "recruiter" in prompt or "asha" in prompt:
        return "recruiter"
    if "hr" in prompt:
        return "hr"
    if "priya" in prompt:
        return "beta_pm"
    if "jordan" in prompt:
        return "client"
    return "manager"


def _read_id(prompt: str) -> str:
    if "ceo" in prompt:
        return "email_019"
    if "build" in prompt:
        return "email_042"
    if "clouddesk" in prompt or "support" in prompt:
        return "email_022"
    if "standup" in prompt:
        return "email_049"
    if "beta" in prompt:
        return "email_038"
    return "email_001"


def _query(prompt: str) -> str:
    if "hotel cancellation" in prompt or "cancellation invoice" in prompt:
        return "Harbor View"
    keywords = [
        "Project Alpha",
        "Project Beta",
        "Amazon",
        "benefits",
        "Boston",
        "reimbursement",
        "Northwind",
        "CloudDesk",
        "Harbor View",
        "onboarding",
        "payment status",
        "newsletter",
        "finance",
    ]
    lower = prompt.lower()
    for keyword in keywords:
        if keyword.lower() in lower:
            return keyword
    return prompt[:80]


def _content_matches(response: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lower = response.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


if __name__ == "__main__":
    raise SystemExit(main())
