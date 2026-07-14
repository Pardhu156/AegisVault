"""Sentinel risk fusion."""

from __future__ import annotations

import statistics

from aegisvault.sentinel.models import MonitorResult, SentinelConfig


def fuse_risk(
    *,
    reasoning: MonitorResult,
    intent: MonitorResult,
    action: MonitorResult,
    config: SentinelConfig,
) -> tuple[float, float, dict[str, float]]:
    """Fuse available monitor drift scores and return risk, confidence, weights."""

    weighted = []
    if reasoning.available and reasoning.drift is not None:
        weighted.append(("reasoning", reasoning.drift, config.reasoning_weight))
    if intent.available and intent.drift is not None:
        weighted.append(("intent", intent.drift, config.intent_weight))
    if action.available and action.drift is not None:
        weighted.append(("action", action.drift, config.action_weight))
    if not weighted:
        return 0.0, 0.0, {}
    total_weight = sum(weight for _, _, weight in weighted)
    normalized = {name: weight / total_weight for name, _, weight in weighted if total_weight > 0}
    fused = sum(drift * normalized[name] for name, drift, _ in weighted)
    confidence = _confidence([drift for _, drift, _ in weighted], len(weighted))
    return max(0.0, min(1.0, fused)), confidence, normalized


def _confidence(drifts: list[float], available_count: int) -> float:
    coverage = available_count / 3.0
    if available_count <= 1:
        agreement = 0.5
    else:
        spread = statistics.pstdev(drifts)
        agreement = max(0.0, 1.0 - min(1.0, spread))
    return max(0.0, min(1.0, 0.65 * coverage + 0.35 * agreement))
