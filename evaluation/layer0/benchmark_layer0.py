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

from aegisvault.layer0 import Layer0Validator
from aegisvault.policy.models import (
    ApplicationConfig,
    DomainPolicy,
    EvaluatorConfig,
    GateConfig,
    GatesConfig,
    Layer0Config,
    Layer0RequestConfig,
    Layer0ToolsConfig,
    LowConfidenceAction,
)


def main() -> int:
    validator = Layer0Validator(policy=_policy())
    samples: list[float] = []
    for index in range(1000):
        started = time.perf_counter()
        validator.validate_request(session_id=f"s-{index}", request_text="Summarize unread email", domain="email")
        validator.validate_tool_call(session_id=f"s-{index}", tool_name="read_email", arguments={"id": "email_1"}, domain="email")
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


def _policy() -> DomainPolicy:
    gate = GateConfig(allow_threshold=0.8, block_threshold=0.8, low_confidence_action=LowConfidenceAction.BLOCK)
    return DomainPolicy(
        version="1.0",
        application=ApplicationConfig(name="layer0-benchmark", description="Layer 0 benchmark"),
        purpose="Benchmark deterministic Layer 0 validation.",
        allowed_topics=["email"],
        gates=GatesConfig(request=gate, response=gate),
        evaluator=EvaluatorConfig(provider="ollama", model="llama3.2"),
        layer0=Layer0Config(
            enabled=True,
            request=Layer0RequestConfig(require_session_id=True, require_domain=True, allowed_domains=["email"]),
            tools=Layer0ToolsConfig(allowlist_mode=True, allowed=["read_email"]),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
