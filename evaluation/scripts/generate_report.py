"""Regenerate metrics and markdown summary for an existing evaluation run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.scripts.eval_lib import build_latency_summary, build_metrics, read_jsonl, write_json
from evaluation.scripts.run_evaluation import _write_markdown_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 evaluation report files.")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    for file_name in ["request_gate_results.jsonl", "response_gate_results.jsonl", "case_results.jsonl"]:
        for row in read_jsonl(args.run_dir / file_name):
            if file_name == "case_results.jsonl" and row.get("mode") in {"request", "response"}:
                continue
            records.append(row)
    metadata_path = args.run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = build_metrics(records)
    latency = build_latency_summary(records)
    write_json(args.run_dir / "metrics.json", metrics)
    write_json(args.run_dir / "latency_summary.json", latency)
    _write_markdown_summary(args.run_dir, metadata, metrics, latency)
    print(f"Regenerated report in {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
