from __future__ import annotations

import pytest

from aegisvault.runtime.goal_vault.embedding import GoalEmbedder
from aegisvault.sentinel import (
    ActionMonitor,
    EmaDriftTracker,
    MonitorResult,
    SentinelConfig,
    SentinelDecisionLevel,
    SentinelExecutionState,
    SentinelMonitor,
    ToolCallState,
    fuse_risk,
)


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-sentinel"
    dimension = 3

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        lower = text.lower()
        if "weather" in lower or "delete" in lower or "stock" in lower:
            return (0.0, 1.0, 0.0)
        if "random" in lower:
            return (0.0, 0.0, 1.0)
        return (1.0, 0.0, 0.0)


def monitor() -> SentinelMonitor:
    return SentinelMonitor(embedder=FakeEmbedder())


def test_goal_equals_intent_low_drift() -> None:
    decision = monitor().analyze(
        session_id="s1",
        trusted_goal="Summarize unread customer emails",
        execution=SentinelExecutionState(current_intent="Summarize unread customer emails"),
    )
    assert decision.intent_similarity == pytest.approx(1.0)
    assert decision.intent_drift == pytest.approx(0.0)
    assert decision.decision == SentinelDecisionLevel.ALLOW


def test_different_intent_high_drift() -> None:
    decision = monitor().analyze(
        session_id="s1",
        trusted_goal="Summarize unread customer emails",
        execution=SentinelExecutionState(current_intent="Check weather forecast"),
    )
    assert decision.intent_drift == pytest.approx(1.0)
    assert decision.decision == SentinelDecisionLevel.BLOCK


def test_reasoning_unavailable_still_functions() -> None:
    decision = monitor().analyze(
        session_id="s1",
        trusted_goal="Summarize unread customer emails",
        execution=SentinelExecutionState(reasoning=None, current_intent="Summarize unread customer emails"),
    )
    assert decision.reasoning_similarity is None
    assert decision.intent_similarity is not None
    assert "reasoning" not in decision.metadata["available_monitors"]


def test_action_normalization() -> None:
    action = ActionMonitor(FakeEmbedder()).normalize_action(ToolCallState(name="delete_email", arguments={"id": 5}))
    assert action == 'delete_email({"id": 5})'


def test_fusion_ignores_unavailable_monitors_and_renormalizes() -> None:
    unavailable = MonitorResult(similarity=None, drift=None, available=False, reason="missing")
    intent = MonitorResult(similarity=0.5, drift=0.5, available=True, reason="intent")
    action = MonitorResult(similarity=0.2, drift=0.8, available=True, reason="action")
    risk, confidence, weights = fuse_risk(reasoning=unavailable, intent=intent, action=action, config=SentinelConfig())
    assert weights["intent"] == pytest.approx(0.35 / 0.80)
    assert weights["action"] == pytest.approx(0.45 / 0.80)
    assert risk == pytest.approx(0.66875)
    assert 0.0 < confidence < 1.0


def test_ema_increases_and_decreases() -> None:
    tracker = EmaDriftTracker(alpha=0.4)
    assert tracker.update("s1", 0.2) == pytest.approx(0.2)
    increased = tracker.update("s1", 1.0)
    assert increased > 0.2
    decreased = tracker.update("s1", 0.0)
    assert decreased < increased


def test_confidence_decreases_when_only_one_monitor_contributes() -> None:
    full = monitor().analyze(
        session_id="s1",
        trusted_goal="Summarize unread customer emails",
        execution=SentinelExecutionState(
            reasoning="Summarize unread customer emails",
            current_intent="Summarize unread customer emails",
            tool_call=ToolCallState(name="read_email", arguments={"folder": "inbox"}),
        ),
    )
    one = monitor().analyze(
        session_id="s1",
        trusted_goal="Summarize unread customer emails",
        execution=SentinelExecutionState(current_intent="Summarize unread customer emails"),
    )
    assert one.confidence < full.confidence


def test_session_isolation() -> None:
    service = monitor()
    high = SentinelExecutionState(current_intent="Check weather forecast")
    low = SentinelExecutionState(current_intent="Summarize unread customer emails")
    service.analyze(session_id="s1", trusted_goal="Summarize unread customer emails", execution=high)
    s2 = service.analyze(session_id="s2", trusted_goal="Summarize unread customer emails", execution=low)
    assert s2.ema_risk == pytest.approx(0.0)


def test_threshold_decisions() -> None:
    config = SentinelConfig(allow_threshold=0.25, observe_threshold=0.45, review_threshold=0.65)
    service = SentinelMonitor(embedder=FakeEmbedder(), config=config)
    assert service.analyze(session_id="a", trusted_goal="goal", execution=SentinelExecutionState(current_intent="goal")).decision == SentinelDecisionLevel.ALLOW
    assert service.analyze(session_id="b", trusted_goal="goal", execution=SentinelExecutionState(current_intent="weather")).decision == SentinelDecisionLevel.BLOCK


def test_embedding_model_loaded_once_by_injection() -> None:
    embedder = FakeEmbedder()
    service = SentinelMonitor(embedder=embedder)
    service.analyze(session_id="s1", trusted_goal="goal", execution=SentinelExecutionState(current_intent="goal"))
    service.analyze(session_id="s1", trusted_goal="goal", execution=SentinelExecutionState(current_intent="goal"))
    assert service.embedder is embedder
    assert embedder.calls == 4
