from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvault.runtime.goal_vault.embedding import GoalEmbedder
from aegisvault.sentinel import SentinelExecutionState, SentinelMonitor, ToolCallState


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-sentinel-benchmark"
    dimension = 2

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0) if "email" in text.lower() else (0.0, 1.0)


def main() -> int:
    sentinel = SentinelMonitor(embedder=FakeEmbedder())
    samples: list[float] = []
    execution = SentinelExecutionState(
        reasoning="Need to summarize unread email.",
        current_intent="Summarize unread email.",
        tool_call=ToolCallState(name="read_email", arguments={"folder": "inbox"}),
    )
    for index in range(1000):
        started = time.perf_counter()
        sentinel.analyze(session_id=f"s-{index % 10}", trusted_goal="Summarize unread email.", execution=execution)
        samples.append((time.perf_counter() - started) * 1000)
    print(
        {
            "iterations": len(samples),
            "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
