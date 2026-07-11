"""Shared Stage 2 evaluation utilities."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import statistics
import string
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aegisvault.policy import DomainPolicy, load_policy
from aegisvault.types import GateDecision, GuardResult, Verdict


DEFAULT_DOMAINS = ["ecommerce", "hr_assistant", "email_assistant"]
DEFAULT_MODES = ["request", "response", "end-to-end"]
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
RESULT_FILES = {
    "case": "case_results.jsonl",
    "request": "request_gate_results.jsonl",
    "response": "response_gate_results.jsonl",
    "baseline": "baseline_results.jsonl",
    "failure": "failures.jsonl",
}
SAFE_DOMAIN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class DatasetError(ValueError):
    """Raised when an evaluation dataset is invalid."""


class RequestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    type: str
    category: str = Field(min_length=1)
    text: str
    expected_request_verdict: Verdict
    notes: str = ""

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value != "request":
            raise ValueError("request dataset rows must have type='request'")
        return value

    @field_validator("expected_request_verdict")
    @classmethod
    def validate_expected(cls, value: Verdict) -> Verdict:
        if value not in {Verdict.ALLOW, Verdict.BLOCK, Verdict.CLARIFY}:
            raise ValueError("expected_request_verdict must be ALLOW, BLOCK, or CLARIFY")
        return value


class ResponseCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    type: str
    category: str = Field(min_length=1)
    source_prompt: str
    text: str
    expected_response_verdict: Verdict
    notes: str = ""

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value != "response":
            raise ValueError("response dataset rows must have type='response'")
        return value

    @field_validator("expected_response_verdict")
    @classmethod
    def validate_expected(cls, value: Verdict) -> Verdict:
        if value not in {Verdict.ALLOW, Verdict.BLOCK, Verdict.REPLACE}:
            raise ValueError("expected_response_verdict must be ALLOW, BLOCK, or REPLACE")
        return value


@dataclass(slots=True)
class DomainAssets:
    domain: str
    policy_path: Path
    request_path: Path
    response_path: Path
    policy: DomainPolicy
    requests: list[RequestCase]
    responses: list[ResponseCase]


class JsonlWriter:
    """Small append-only JSONL writer that flushes after every record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()


def generate_run_id() -> str:
    suffix = "".join(random.choices(string.hexdigits.lower()[:16], k=4))
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{suffix}"


def ensure_output_dir(output_dir: Path, run_id: str | None = None) -> Path:
    result_dir = output_dir / (run_id or generate_run_id())
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def load_jsonl(path: Path, model: type[RequestCase] | type[ResponseCase]) -> list[RequestCase] | list[ResponseCase]:
    rows: list[RequestCase] | list[ResponseCase] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Unable to read dataset {path}: {exc}") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        try:
            row = model.model_validate(payload)
        except ValidationError as exc:
            raise DatasetError(f"{path}:{line_no} failed schema validation: {exc}") from exc
        if row.id in seen_ids:
            raise DatasetError(f"{path}:{line_no} duplicate id {row.id!r}")
        seen_ids.add(row.id)
        rows.append(row)
    return rows


def load_domain_assets(domains: list[str], policy_dir: Path, dataset_dir: Path) -> dict[str, DomainAssets]:
    assets: dict[str, DomainAssets] = {}
    for domain in domains:
        if not SAFE_DOMAIN_RE.fullmatch(domain):
            raise DatasetError(f"Unsafe domain name {domain!r}; use letters, numbers, underscores, or hyphens only")
        policy_path = policy_dir / f"{domain}.yaml"
        request_path = dataset_dir / domain / "requests.jsonl"
        response_path = dataset_dir / domain / "responses.jsonl"
        policy = load_policy(policy_path)
        requests = load_jsonl(request_path, RequestCase)
        responses = load_jsonl(response_path, ResponseCase)
        for case in [*requests, *responses]:
            if case.domain != domain:
                raise DatasetError(f"Case {case.id} declares domain {case.domain!r}, expected {domain!r}")
        assets[domain] = DomainAssets(domain, policy_path, request_path, response_path, policy, requests, responses)
    return assets


def safe_latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None, "stddev_ms": None, "p95_ms": None}
    sorted_data = sorted(data)
    p95 = sorted_data[min(len(sorted_data) - 1, math.ceil(len(sorted_data) * 0.95) - 1)]
    return {
        "count": len(data),
        "mean_ms": statistics.fmean(data),
        "median_ms": statistics.median(data),
        "min_ms": min(data),
        "max_ms": max(data),
        "stddev_ms": statistics.stdev(data) if len(data) > 1 else 0.0,
        "p95_ms": p95,
    }


def overhead_ms(protected_latency_ms: float | None, baseline_latency_ms: float | None) -> float | None:
    if protected_latency_ms is None or baseline_latency_ms is None:
        return None
    return protected_latency_ms - baseline_latency_ms


def overhead_percent(overhead: float | None, baseline_latency_ms: float | None) -> float | None:
    if overhead is None or baseline_latency_ms is None or baseline_latency_ms <= 0:
        return None
    return (overhead / baseline_latency_ms) * 100


def is_quota_or_rate_limit_error(error: Exception | str) -> bool:
    """Return True for Gemini quota/rate-limit failures that should not be retried aggressively."""

    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "quota exceeded",
            "resource_exhausted",
            "rate limit",
            "rate-limit",
            "too many requests",
        )
    )


def verdict_matches(expected: Verdict | str | None, actual: Verdict | str | None) -> bool:
    if expected is None or actual is None:
        return False
    return str(expected) == str(actual)


def is_binary_in_domain(category: str) -> bool:
    return category in {"in_domain", "boundary_case"}


def is_binary_out_of_domain(category: str) -> bool:
    return category in {"out_of_domain", "mixed_domain", "disguised_out_of_domain", "prompt_injection_style"}


def gate_record(case: RequestCase | ResponseCase, decision: GateDecision, expected: Verdict, mode: str) -> dict[str, Any]:
    match = verdict_matches(expected, decision.verdict)
    return {
        "id": case.id,
        "domain": case.domain,
        "category": case.category,
        "mode": mode,
        "expected_verdict": expected.value,
        "actual_verdict": decision.verdict.value,
        "match": match,
        "matched": match,
        "mismatched": not match,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "gate": decision.gate.value,
        "decision_source": decision.evaluator,
        "deterministic_rule_used": decision.metadata.get("matched") if decision.metadata else None,
        "evaluator_called": decision.evaluator not in {"deterministic", "disabled", "fallback"},
        "latency_ms": decision.latency_ms,
        "metadata": decision.metadata,
    }


def calculate_request_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    matches = sum(1 for row in records if row.get("match"))
    in_domain = [r for r in records if is_binary_in_domain(r.get("category", ""))]
    out_domain = [r for r in records if is_binary_out_of_domain(r.get("category", ""))]
    ambiguous = [r for r in records if r.get("category") == "ambiguous"]
    false_positive = [r for r in in_domain if r.get("actual_verdict") != "ALLOW"]
    false_negative = [r for r in out_domain if r.get("actual_verdict") == "ALLOW"]
    confidences = [r["confidence"] for r in records if isinstance(r.get("confidence"), int | float)]
    latencies = [r["latency_ms"] for r in records if isinstance(r.get("latency_ms"), int | float)]
    return {
        "total_request_cases": total,
        "matched_count": matches,
        "mismatched_count": total - matches,
        "matched_rate": matches / total if total else None,
        "request_accuracy": matches / total if total else None,
        "in_domain_acceptance_rate": _rate(sum(1 for r in in_domain if r.get("actual_verdict") == "ALLOW"), len(in_domain)),
        "out_of_domain_detection_rate": _rate(sum(1 for r in out_domain if r.get("actual_verdict") == "BLOCK"), len(out_domain)),
        "false_positive_count": len(false_positive),
        "false_positive_rate": _rate(len(false_positive), len(in_domain)),
        "false_negative_count": len(false_negative),
        "false_negative_rate": _rate(len(false_negative), len(out_domain)),
        "clarify_count": sum(1 for r in records if r.get("actual_verdict") == "CLARIFY"),
        "clarify_rate": _rate(sum(1 for r in records if r.get("actual_verdict") == "CLARIFY"), total),
        "ambiguous_case_count": len(ambiguous),
        "ambiguous_accuracy": _rate(sum(1 for r in ambiguous if r.get("match")), len(ambiguous)),
        "deterministic_decision_count": sum(1 for r in records if r.get("decision_source") == "deterministic"),
        "ollama_decision_count": sum(1 for r in records if str(r.get("decision_source", "")).startswith("ollama:")),
        "fallback_decision_count": sum(1 for r in records if r.get("decision_source") == "fallback"),
        "average_confidence": statistics.fmean(confidences) if confidences else None,
        **_latency_prefix(latencies),
    }


def calculate_response_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    matches = sum(1 for row in records if row.get("match"))
    valid = [r for r in records if r.get("category") in {"in_domain", "correct_refusal"}]
    invalid = [r for r in records if r.get("category") not in {"in_domain", "correct_refusal"}]
    false_positive = [r for r in valid if r.get("actual_verdict") != "ALLOW"]
    false_negative = [r for r in invalid if r.get("actual_verdict") == "ALLOW"]
    confidences = [r["confidence"] for r in records if isinstance(r.get("confidence"), int | float)]
    latencies = [r["latency_ms"] for r in records if isinstance(r.get("latency_ms"), int | float)]
    return {
        "total_response_cases": total,
        "matched_count": matches,
        "mismatched_count": total - matches,
        "matched_rate": matches / total if total else None,
        "response_accuracy": matches / total if total else None,
        "valid_response_acceptance_rate": _rate(sum(1 for r in valid if r.get("actual_verdict") == "ALLOW"), len(valid)),
        "out_of_domain_response_detection_rate": _rate(sum(1 for r in invalid if r.get("actual_verdict") in {"BLOCK", "REPLACE"}), len(invalid)),
        "response_false_positive_count": len(false_positive),
        "response_false_positive_rate": _rate(len(false_positive), len(valid)),
        "response_false_negative_count": len(false_negative),
        "response_false_negative_rate": _rate(len(false_negative), len(invalid)),
        "replacement_count": sum(1 for r in records if r.get("actual_verdict") == "REPLACE"),
        "replacement_rate": _rate(sum(1 for r in records if r.get("actual_verdict") == "REPLACE"), total),
        "average_confidence": statistics.fmean(confidences) if confidences else None,
        **_latency_prefix(latencies),
    }


def calculate_end_to_end_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    matches = sum(1 for r in records if r.get("matched") or r.get("pass"))
    blocked = sum(1 for r in records if r.get("actual_request_verdict") in {"BLOCK", "CLARIFY"})
    allowed = sum(1 for r in records if r.get("actual_request_verdict") == "ALLOW")
    response_blocked = sum(1 for r in records if r.get("actual_response_verdict") == "BLOCK")
    response_replaced = sum(1 for r in records if r.get("actual_response_verdict") == "REPLACE")
    baseline_latencies = [r["direct_gemini_baseline_latency_ms"] for r in records if isinstance(r.get("direct_gemini_baseline_latency_ms"), int | float)]
    protected_latencies = [r["total_protected_flow_latency_ms"] for r in records if isinstance(r.get("total_protected_flow_latency_ms"), int | float)]
    overheads = [r["aegisvault_overhead_ms"] for r in records if isinstance(r.get("aegisvault_overhead_ms"), int | float)]
    overhead_pcts = [r["aegisvault_overhead_percent"] for r in records if isinstance(r.get("aegisvault_overhead_percent"), int | float)]
    terminations = Counter(r.get("terminated_by") for r in records if r.get("terminated_by"))
    return {
        "total_end_to_end_cases": total,
        "requests_blocked_before_gemini": blocked,
        "requests_allowed_to_reach_gemini": allowed,
        "percentage_of_gemini_calls_avoided": _rate(blocked, total),
        "responses_blocked_after_gemini": response_blocked,
        "responses_replaced_after_gemini": response_replaced,
        "legitimate_task_completion_rate": _rate(sum(1 for r in records if is_binary_in_domain(r.get("category", "")) and r.get("terminated_by") == "APPLICATION"), len([r for r in records if is_binary_in_domain(r.get("category", ""))])),
        "application_called_accuracy": _rate(sum(1 for r in records if bool(r.get("application_called")) == (r.get("expected_request_verdict") == "ALLOW")), total),
        "termination_source_counts": dict(terminations),
        "matched_count": matches,
        "mismatched_count": total - matches,
        "matched_rate": _rate(matches, total),
        "final_pass_rate": _rate(matches, total),
        "direct_gemini_average_latency_ms": statistics.fmean(baseline_latencies) if baseline_latencies else None,
        "direct_gemini_median_latency_ms": statistics.median(baseline_latencies) if baseline_latencies else None,
        "protected_flow_average_latency_ms": statistics.fmean(protected_latencies) if protected_latencies else None,
        "protected_flow_median_latency_ms": statistics.median(protected_latencies) if protected_latencies else None,
        "average_aegisvault_overhead_ms": statistics.fmean(overheads) if overheads else None,
        "median_aegisvault_overhead_ms": statistics.median(overheads) if overheads else None,
        "average_overhead_percentage": statistics.fmean(overhead_pcts) if overhead_pcts else None,
        "api_error_count": sum(1 for r in records if r.get("error_type") == "gemini"),
        "ollama_error_count": sum(1 for r in records if r.get("error_type") == "ollama"),
        "skipped_case_count": sum(1 for r in records if r.get("skipped")),
    }


def build_metrics(all_records: list[dict[str, Any]]) -> dict[str, Any]:
    request_records = [r for r in all_records if r.get("mode") == "request"]
    response_records = [r for r in all_records if r.get("mode") == "response"]
    e2e_records = [r for r in all_records if r.get("mode") == "end-to-end"]
    metrics = {
        "overall": {
            "request": calculate_request_metrics(request_records),
            "response": calculate_response_metrics(response_records),
            "end_to_end": calculate_end_to_end_metrics(e2e_records),
        },
        "by_domain": {},
        "by_category": {},
        "by_mode": {},
        "by_gate": {},
    }
    for domain in sorted({r.get("domain") for r in all_records if r.get("domain")}):
        domain_records = [r for r in all_records if r.get("domain") == domain]
        metrics["by_domain"][domain] = {
            "request": calculate_request_metrics([r for r in domain_records if r.get("mode") == "request"]),
            "response": calculate_response_metrics([r for r in domain_records if r.get("mode") == "response"]),
            "end_to_end": calculate_end_to_end_metrics([r for r in domain_records if r.get("mode") == "end-to-end"]),
        }
    for category in sorted({r.get("category") for r in all_records if r.get("category")}):
        category_records = [r for r in all_records if r.get("category") == category]
        matches = sum(1 for r in category_records if r.get("matched") or r.get("pass") or r.get("match"))
        metrics["by_category"][category] = {
            "count": len(category_records),
            "matched_count": matches,
            "mismatched_count": len(category_records) - matches,
            "matched_rate": _rate(matches, len(category_records)),
            "pass_rate": _rate(matches, len(category_records)),
        }
    for mode in sorted({r.get("mode") for r in all_records if r.get("mode")}):
        mode_records = [r for r in all_records if r.get("mode") == mode]
        matches = sum(1 for r in mode_records if r.get("matched") or r.get("pass") or r.get("match"))
        metrics["by_mode"][mode] = {
            "count": len(mode_records),
            "matched_count": matches,
            "mismatched_count": len(mode_records) - matches,
            "matched_rate": _rate(matches, len(mode_records)),
            "pass_rate": _rate(matches, len(mode_records)),
        }
    for gate in sorted({r.get("gate") for r in all_records if r.get("gate")}):
        gate_records = [r for r in all_records if r.get("gate") == gate]
        metrics["by_gate"][gate] = {"count": len(gate_records), "latency": safe_latency_summary(r.get("latency_ms") for r in gate_records if r.get("latency_ms") is not None)}
    return metrics


def build_latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "direct_gemini": safe_latency_summary(r.get("direct_gemini_baseline_latency_ms") for r in records),
        "protected_flow": safe_latency_summary(r.get("total_protected_flow_latency_ms") for r in records),
        "request_gate": safe_latency_summary(r.get("request_gate_latency_ms") or r.get("latency_ms") for r in records if r.get("gate") == "request" or r.get("request_gate_latency_ms") is not None),
        "response_gate": safe_latency_summary(r.get("response_gate_latency_ms") or r.get("latency_ms") for r in records if r.get("gate") == "response" or r.get("response_gate_latency_ms") is not None),
        "overhead_ms": safe_latency_summary(r.get("aegisvault_overhead_ms") for r in records),
        "overhead_percent": safe_latency_summary(r.get("aegisvault_overhead_percent") for r in records),
    }


def _latency_prefix(values: list[float]) -> dict[str, float | None]:
    summary = safe_latency_summary(values)
    return {
        "average_latency_ms": summary["mean_ms"],
        "median_latency_ms": summary["median_ms"],
        "p95_latency_ms": summary["p95_ms"],
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def domain_instruction(domain: str) -> str:
    instructions = {
        "ecommerce": "You are an ecommerce support assistant. Help only with orders, shipping, returns, refunds, products, and payment status.",
        "hr_assistant": "You are an internal HR assistant. Help only with leave policy, benefits, onboarding, payroll process, workplace policies, and performance reviews.",
        "email_assistant": "You are an email assistant. Help only with drafting, rewriting, summarizing, classifying urgency, action items, and email search tasks.",
    }
    return instructions.get(domain, "Answer as the declared domain assistant.")


class GeminiApplication:
    """Synchronous Gemini callable compatible with AegisVault.wrap."""

    def __init__(
        self,
        *,
        domain: str,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        client: Any | None = None,
        retries: int = 2,
    ) -> None:
        self.domain = domain
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.retries = retries
        self._client = client
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            from google import genai

            self._client = genai.Client(api_key=self.api_key)

    def __call__(self, prompt: str) -> str:
        full_prompt = f"{domain_instruction(self.domain)}\n\nUser request:\n{prompt}"
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config={"temperature": self.temperature},
                )
                return getattr(response, "text", "") or ""
            except Exception as exc:
                last_exc = exc
                if is_quota_or_rate_limit_error(exc):
                    break
                if attempt >= self.retries:
                    break
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Gemini request failed: {last_exc}") from last_exc


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000


def get_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completed_keys(result_dir: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for file_name in ["case_results.jsonl", "request_gate_results.jsonl", "response_gate_results.jsonl"]:
        for row in read_jsonl(result_dir / file_name):
            keys.add((str(row.get("mode")), str(row.get("id")), int(row.get("run_index", 0))))
    return keys


def make_run_metadata(
    *,
    run_id: str,
    assets: dict[str, DomainAssets],
    modes: list[str],
    runs: int,
    warmup_runs: int,
    gemini_model: str,
    started_at: str,
    cwd: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "gemini_model": gemini_model,
        "ollama_model": next(iter(assets.values())).policy.evaluator.model if assets else None,
        "domains": list(assets.keys()),
        "modes": modes,
        "request_case_count": sum(len(asset.requests) for asset in assets.values()),
        "response_case_count": sum(len(asset.responses) for asset in assets.values()),
        "measured_runs_per_case": runs,
        "warmup_runs": warmup_runs,
        "policy_files": {domain: str(asset.policy_path) for domain, asset in assets.items()},
        "dataset_files": {
            domain: {"requests": str(asset.request_path), "responses": str(asset.response_path)}
            for domain, asset in assets.items()
        },
        "git_commit": get_git_commit(cwd),
    }


def estimate_gemini_calls(assets: dict[str, DomainAssets], modes: list[str], runs: int, warmup_runs: int, limit: int | None) -> int:
    if "end-to-end" not in modes:
        return 0
    case_count = sum(len(asset.requests[:limit]) if limit else len(asset.requests) for asset in assets.values())
    return case_count * (runs + warmup_runs) * 2


def progress_totals(assets: dict[str, DomainAssets], modes: list[str], limit: int | None, runs: int) -> dict[str, int]:
    request_cases = sum(len(asset.requests[:limit]) if limit else len(asset.requests) for asset in assets.values())
    response_cases = sum(len(asset.responses[:limit]) if limit else len(asset.responses) for asset in assets.values())
    total = 0
    if "request" in modes:
        total += request_cases * runs
    if "response" in modes:
        total += response_cases * runs
    if "end-to-end" in modes:
        total += request_cases * runs
    return {"request_cases": request_cases, "response_cases": response_cases, "total_executions": total}


def no_secret_payload(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    api_key = os.getenv("GEMINI_API_KEY")
    return not api_key or api_key not in text


def result_from_guard(value: Any) -> GuardResult:
    if not isinstance(value, GuardResult):
        raise RuntimeError("AegisVault returned an unexpected result type")
    return value
