"""Run Stage 2 AegisVault evaluation with Gemini as the protected app."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tqdm.auto import tqdm

from aegisvault import AegisVault
from aegisvault.gates import RequestGate, ResponseGate
from aegisvault.types import EvaluationContext

from evaluation.scripts.eval_lib import (
    DEFAULT_DOMAINS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODES,
    GeminiApplication,
    JsonlWriter,
    RequestCase,
    ResponseCase,
    RESULT_FILES,
    build_latency_summary,
    build_metrics,
    completed_keys,
    ensure_output_dir,
    estimate_gemini_calls,
    gate_record,
    make_run_metadata,
    no_secret_payload,
    overhead_ms,
    overhead_percent,
    is_quota_or_rate_limit_error,
    progress_totals,
    read_jsonl,
    result_from_guard,
    time_call,
    write_json,
    load_domain_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AegisVault Stage 2 evaluation.")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=DEFAULT_MODES)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per case")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit cases per domain and dataset")
    parser.add_argument("--policy-dir", type=Path, default=Path("evaluation/policies"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("evaluation/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Existing run folder to resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")

    assets = load_domain_assets(args.domains, args.policy_dir, args.dataset_dir)
    totals = progress_totals(assets, args.modes, args.limit, args.runs)
    estimated_gemini_calls = estimate_gemini_calls(assets, args.modes, args.runs, args.warmup_runs, args.limit)

    print("Evaluation plan")
    print(f"Domains: {len(assets)}")
    print(f"Request cases: {totals['request_cases']}")
    print(f"Response cases: {totals['response_cases']}")
    print(f"Evaluation modes: {', '.join(args.modes)}")
    print(f"Measured runs per case: {args.runs}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Estimated Gemini calls: {estimated_gemini_calls}")

    started_at = datetime.now(UTC).isoformat()
    if args.resume:
        result_dir = args.resume
        result_dir.mkdir(parents=True, exist_ok=True)
        run_id = result_dir.name
    else:
        result_dir = ensure_output_dir(args.output_dir)
        run_id = result_dir.name

    metadata = make_run_metadata(
        run_id=run_id,
        assets=assets,
        modes=args.modes,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        gemini_model=args.gemini_model,
        started_at=started_at,
        cwd=Path.cwd(),
    )
    write_json(result_dir / "run_metadata.json", metadata)

    writers = {name: JsonlWriter(result_dir / file_name) for name, file_name in RESULT_FILES.items()}
    done = completed_keys(result_dir) if args.resume else set()
    counters = {
        "matched": 0,
        "mismatched": 0,
        "skipped": 0,
        "gemini_calls": 0,
        "ollama_calls": 0,
        "latency_total_ms": 0.0,
        "latency_count": 0,
    }
    run_started = time.perf_counter()

    with tqdm(total=totals["total_executions"], dynamic_ncols=True) as bar:
        for domain, asset in assets.items():
            request_cases = asset.requests[: args.limit] if args.limit else asset.requests
            response_cases = asset.responses[: args.limit] if args.limit else asset.responses

            if "request" in args.modes:
                gate = RequestGate(asset.policy, AegisVault(policy=asset.policy).evaluator)
                for case in request_cases:
                    for run_index in range(args.runs):
                        key = ("request", case.id, run_index)
                        if key in done:
                            counters["skipped"] += 1
                            bar.update(1)
                            continue
                        _run_request_case(case, gate, run_index, writers, counters, args.fail_fast)
                        _update_bar(bar, domain, "request", counters)

            if "response" in args.modes:
                gate = ResponseGate(asset.policy, AegisVault(policy=asset.policy).evaluator)
                for case in response_cases:
                    for run_index in range(args.runs):
                        key = ("response", case.id, run_index)
                        if key in done:
                            counters["skipped"] += 1
                            bar.update(1)
                            continue
                        _run_response_case(case, gate, run_index, writers, counters, args.fail_fast)
                        _update_bar(bar, domain, "response", counters)

            if "end-to-end" in args.modes:
                app = GeminiApplication(domain=domain, model=args.gemini_model)
                timed_app = _TimedCallable(app)
                guard = AegisVault.from_policy(asset.policy_path)
                protected_app = guard.wrap(timed_app)
                for case in request_cases:
                    for warmup_index in range(args.warmup_runs):
                        _run_e2e_case(case, protected_app, app, timed_app, warmup_index, True, writers, counters, args.fail_fast, verbose_case=False)
                    for run_index in range(args.runs):
                        key = ("end-to-end", case.id, run_index)
                        if key in done:
                            counters["skipped"] += 1
                            bar.update(1)
                            continue
                        _run_e2e_case(case, protected_app, app, timed_app, run_index, False, writers, counters, args.fail_fast, verbose_case=args.limit is not None)
                        _update_bar(bar, domain, "end-to-end", counters)

    all_records = _load_records_for_metrics(result_dir)
    metrics = build_metrics(all_records)
    metrics["overall"]["end_to_end"]["total_evaluation_runtime_seconds"] = time.perf_counter() - run_started
    metrics["overall"]["end_to_end"]["total_gemini_calls"] = counters["gemini_calls"]
    metrics["overall"]["end_to_end"]["total_ollama_evaluator_calls"] = counters["ollama_calls"]
    latency_summary = build_latency_summary(all_records)

    metadata["completed_at"] = datetime.now(UTC).isoformat()
    write_json(result_dir / "run_metadata.json", metadata)
    write_json(result_dir / "metrics.json", metrics)
    write_json(result_dir / "latency_summary.json", latency_summary)
    _write_markdown_summary(result_dir, metadata, metrics, latency_summary)

    print("\nEvaluation complete")
    print(f"Total cases discovered: {totals['request_cases'] + totals['response_cases']}")
    print(f"Total executions: {totals['total_executions']}")
    print(f"Matched: {counters['matched']}")
    print(f"Mismatched: {counters['mismatched']}")
    print(f"False positives: {_overall_false_positive_count(metrics)}")
    print(f"False negatives: {_overall_false_negative_count(metrics)}")
    print(f"Skipped: {counters['skipped']}")
    print(f"Gemini calls: {counters['gemini_calls']}")
    print(f"Ollama evaluator calls: {counters['ollama_calls']}")
    print(f"Total runtime seconds: {time.perf_counter() - run_started:.2f}")
    print(f"Result folder: {result_dir}")
    return 0 if counters["mismatched"] == 0 else 1


def _run_request_case(
    case: RequestCase,
    gate: RequestGate,
    run_index: int,
    writers: dict[str, JsonlWriter],
    counters: dict[str, Any],
    fail_fast: bool,
) -> None:
    try:
        decision = gate.evaluate(case.text, EvaluationContext(request_text=case.text))
        record = gate_record(case, decision, case.expected_request_verdict, "request")
        record["run_index"] = run_index
        record["pass"] = record["match"]
        writers["request"].write(record)
        writers["case"].write(record)
        counters["ollama_calls"] += int(record["evaluator_called"])
        _record_match(counters, bool(record["match"]), decision.latency_ms)
        if not record["pass"]:
            writers["failure"].write(record)
    except Exception as exc:
        _record_failure(writers, counters, case, "request", run_index, exc, fail_fast)


def _run_response_case(
    case: ResponseCase,
    gate: ResponseGate,
    run_index: int,
    writers: dict[str, JsonlWriter],
    counters: dict[str, Any],
    fail_fast: bool,
) -> None:
    try:
        decision = gate.evaluate(case.text, EvaluationContext(request_text=case.source_prompt, response_text=case.text))
        record = gate_record(case, decision, case.expected_response_verdict, "response")
        record["source_prompt"] = case.source_prompt
        record["run_index"] = run_index
        record["pass"] = record["match"]
        writers["response"].write(record)
        writers["case"].write(record)
        counters["ollama_calls"] += int(record["evaluator_called"])
        _record_match(counters, bool(record["match"]), decision.latency_ms)
        if not record["pass"]:
            writers["failure"].write(record)
    except Exception as exc:
        _record_failure(writers, counters, case, "response", run_index, exc, fail_fast)


def _run_e2e_case(
    case: RequestCase,
    protected_app: Any,
    direct_app: Any,
    timed_app: "_TimedCallable",
    run_index: int,
    warmup: bool,
    writers: dict[str, JsonlWriter],
    counters: dict[str, Any],
    fail_fast: bool,
    *,
    verbose_case: bool,
) -> None:
    try:
        baseline_response: str | None = None
        baseline_latency_ms: float | None = None
        protected_latency_ms: float | None = None
        protected_response = None

        baseline_response, baseline_latency_ms = time_call(lambda: direct_app(case.text))
        counters["gemini_calls"] += 1
        if not warmup:
            writers["baseline"].write(
                {
                    "id": case.id,
                    "domain": case.domain,
                    "category": case.category,
                    "mode": "end-to-end",
                    "run_index": run_index,
                    "direct_gemini_baseline_response": baseline_response,
                    "direct_gemini_baseline_latency_ms": baseline_latency_ms,
                }
            )

        timed_app.last_latency_ms = None
        protected_response, protected_latency_ms = time_call(lambda: result_from_guard(protected_app(case.text)))
        if protected_response.application_called:
            counters["gemini_calls"] += 1
        request_decision = protected_response.request_decision
        response_decision = protected_response.response_decision
        counters["ollama_calls"] += int(request_decision is not None and request_decision.evaluator not in {"deterministic", "disabled", "fallback"})
        counters["ollama_calls"] += int(response_decision is not None and response_decision.evaluator not in {"deterministic", "disabled", "fallback"})

        if warmup:
            return

        overhead = overhead_ms(protected_latency_ms, baseline_latency_ms)
        overhead_pct = overhead_percent(overhead, baseline_latency_ms)
        request_match = request_decision is not None and request_decision.verdict == case.expected_request_verdict
        final_pass = request_match
        record = {
            "id": case.id,
            "domain": case.domain,
            "category": case.category,
            "mode": "end-to-end",
            "run_index": run_index,
            "input_prompt": case.text,
            "expected_request_verdict": case.expected_request_verdict.value,
            "actual_request_verdict": request_decision.verdict.value if request_decision else None,
            "request_gate_confidence": request_decision.confidence if request_decision else None,
            "request_gate_reason": request_decision.reason if request_decision else None,
            "request_gate_latency_ms": request_decision.latency_ms if request_decision else None,
            "gemini_called_through_protected_flow": protected_response.application_called,
            "gemini_protected_flow_response": protected_response.original_response,
            "gemini_protected_flow_latency_ms": timed_app.last_latency_ms,
            "expected_response_verdict": None,
            "actual_response_verdict": response_decision.verdict.value if response_decision else None,
            "response_gate_confidence": response_decision.confidence if response_decision else None,
            "response_gate_reason": response_decision.reason if response_decision else None,
            "response_gate_latency_ms": response_decision.latency_ms if response_decision else None,
            "final_returned_response": protected_response.final_response,
            "terminated_by": protected_response.terminated_by.value,
            "application_called": protected_response.application_called,
            "was_modified": protected_response.was_modified,
            "total_protected_flow_latency_ms": protected_latency_ms,
            "direct_gemini_baseline_response": baseline_response,
            "direct_gemini_baseline_latency_ms": baseline_latency_ms,
            "aegisvault_overhead_ms": overhead,
            "aegisvault_overhead_percent": overhead_pct,
            "matched": final_pass,
            "mismatched": not final_pass,
            "pass": final_pass,
            "skipped": False,
            "error": None,
            "error_type": None,
        }
        if not no_secret_payload(record):
            raise RuntimeError("Refusing to write output because it contains GEMINI_API_KEY")
        writers["case"].write(record)
        if verbose_case:
            print(
                "SMOKE "
                f"{case.domain}/{case.id}: expected={record['expected_request_verdict']} "
                f"actual={record['actual_request_verdict']} gemini_called={record['application_called']} "
                f"response_gate={record['actual_response_verdict']} terminated_by={record['terminated_by']} "
                f"direct_ms={_fmt_ms(record['direct_gemini_baseline_latency_ms'])} "
                f"protected_ms={_fmt_ms(record['total_protected_flow_latency_ms'])} "
                f"overhead_ms={_fmt_ms(record['aegisvault_overhead_ms'])} "
                f"matched={record['matched']}"
            )
        _record_match(counters, final_pass, protected_latency_ms)
        if not final_pass:
            writers["failure"].write(record)
    except Exception as exc:
        _record_failure(writers, counters, case, "end-to-end", run_index, exc, fail_fast, warmup=warmup)


def _record_failure(
    writers: dict[str, JsonlWriter],
    counters: dict[str, Any],
    case: RequestCase | ResponseCase,
    mode: str,
    run_index: int,
    exc: Exception,
    fail_fast: bool,
    *,
    warmup: bool = False,
) -> None:
    if warmup:
        if fail_fast:
            raise exc
        return
    skipped = mode == "end-to-end" and is_quota_or_rate_limit_error(exc)
    record = {
        "id": case.id,
        "domain": case.domain,
        "category": case.category,
        "mode": mode,
        "run_index": run_index,
        "matched": False,
        "mismatched": not skipped,
        "pass": False,
        "skipped": skipped,
        "error": str(exc),
        "error_type": "gemini" if "Gemini" in str(exc) or "GEMINI" in str(exc) else "ollama" if "Ollama" in str(exc) else "runtime",
    }
    writers["failure"].write(record)
    writers["case"].write(record)
    counters["skipped" if skipped else "mismatched"] += 1
    if fail_fast:
        raise exc


class _TimedCallable:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.last_latency_ms: float | None = None

    def __call__(self, prompt: str) -> str:
        result, latency_ms = time_call(lambda: self.app(prompt))
        self.last_latency_ms = latency_ms
        return result


def _update_bar(bar: tqdm, domain: str, mode: str, counters: dict[str, Any]) -> None:
    avg = _current_average_latency(counters)
    bar.set_description(f"{domain} {mode}")
    bar.set_postfix(
        {
            "matched": counters["matched"],
            "mismatched": counters["mismatched"],
            "skipped": counters["skipped"],
            "avg_ms": avg,
        }
    )
    bar.update(1)


def _record_match(counters: dict[str, Any], matched: bool, latency_ms: float | None) -> None:
    counters["matched" if matched else "mismatched"] += 1
    if latency_ms is not None:
        counters["latency_total_ms"] += float(latency_ms)
        counters["latency_count"] += 1


def _current_average_latency(counters: dict[str, Any]) -> str:
    count = int(counters.get("latency_count") or 0)
    if count == 0:
        return "n/a"
    return f"{float(counters['latency_total_ms']) / count:.1f}"


def _overall_false_positive_count(metrics: dict[str, Any]) -> int:
    request = metrics["overall"]["request"].get("false_positive_count") or 0
    response = metrics["overall"]["response"].get("response_false_positive_count") or 0
    return int(request) + int(response)


def _overall_false_negative_count(metrics: dict[str, Any]) -> int:
    request = metrics["overall"]["request"].get("false_negative_count") or 0
    response = metrics["overall"]["response"].get("response_false_negative_count") or 0
    return int(request) + int(response)


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _load_records_for_metrics(result_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_name in ["request_gate_results.jsonl", "response_gate_results.jsonl", "case_results.jsonl"]:
        for row in read_jsonl(result_dir / file_name):
            if file_name == "case_results.jsonl" and row.get("mode") in {"request", "response"}:
                continue
            records.append(row)
    return records


def _write_markdown_summary(result_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any], latency: dict[str, Any]) -> None:
    failures = read_jsonl(result_dir / "failures.jsonl")[:10]
    lines = [
        "# AegisVault Stage 2 Evaluation Summary",
        "",
        "These measurements are from the current machine, network, Gemini model, Ollama model, and labelled dataset. They are not statistically conclusive.",
        "",
        "## Run Configuration",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Gemini model: `{metadata['gemini_model']}`",
        f"- Ollama model: `{metadata['ollama_model']}`",
        f"- Domains: {', '.join(metadata['domains'])}",
        f"- Modes: {', '.join(metadata['modes'])}",
        f"- Measured runs per case: {metadata['measured_runs_per_case']}",
        f"- Warmup runs: {metadata['warmup_runs']}",
        f"- Output folder: `{result_dir}`",
        "",
        "## Overall Metrics",
        "",
        "```json",
        json_dumps(metrics["overall"]),
        "```",
        "",
        "## Latency Summary",
        "",
        "```json",
        json_dumps(latency),
        "```",
        "",
        "## Per-Domain Results",
        "",
        "```json",
        json_dumps(metrics["by_domain"]),
        "```",
        "",
        "## Failure Examples",
        "",
        "```json",
        json_dumps(failures),
        "```",
        "",
        "## Ambiguous-Case Analysis",
        "",
        "Ambiguous request rows are labelled separately and are not folded into binary false-positive or false-negative rates.",
        "",
        "## Limitations",
        "",
        "- The datasets are artificial and manually labelled for evaluation repeatability.",
        "- Gemini responses vary with model availability, network conditions, and provider-side changes.",
        "- Ollama model confidence is model-reported and not statistically calibrated.",
        "- Gemini is the protected application, not the evaluator or ground-truth judge.",
    ]
    (result_dir / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
