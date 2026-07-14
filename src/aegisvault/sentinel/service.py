"""Sentinel runtime monitoring service."""

from __future__ import annotations

from aegisvault.runtime.goal_vault.embedding import GoalEmbedder, SentenceTransformerGoalEmbedder
from aegisvault.sentinel.action_monitor import ActionMonitor
from aegisvault.sentinel.ema import EmaDriftTracker
from aegisvault.sentinel.fusion import fuse_risk
from aegisvault.sentinel.intent_monitor import IntentMonitor
from aegisvault.sentinel.models import SentinelConfig, SentinelDecision, SentinelDecisionLevel, SentinelExecutionState
from aegisvault.sentinel.reasoning_monitor import ReasoningMonitor


class SentinelMonitor:
    """Standalone runtime drift monitor.

    Sentinel does not authorize actions, execute tools, modify Goal Vault, or call
    other gates. It only emits runtime security signals.
    """

    def __init__(
        self,
        *,
        embedder: GoalEmbedder | None = None,
        config: SentinelConfig | None = None,
        reasoning_monitor: ReasoningMonitor | None = None,
        intent_monitor: IntentMonitor | None = None,
        action_monitor: ActionMonitor | None = None,
        ema_tracker: EmaDriftTracker | None = None,
    ) -> None:
        self.config = config or SentinelConfig()
        self.embedder = embedder or SentenceTransformerGoalEmbedder()
        self.reasoning_monitor = reasoning_monitor or ReasoningMonitor(self.embedder)
        self.intent_monitor = intent_monitor or IntentMonitor(self.embedder)
        self.action_monitor = action_monitor or ActionMonitor(self.embedder)
        self.ema_tracker = ema_tracker or EmaDriftTracker(alpha=self.config.ema_alpha)

    def analyze(
        self,
        *,
        session_id: str,
        trusted_goal: str,
        execution: SentinelExecutionState,
    ) -> SentinelDecision:
        """Analyze a structured execution state against a trusted goal."""

        reasoning = self.reasoning_monitor.evaluate(trusted_goal=trusted_goal, reasoning=execution.reasoning)
        intent = self.intent_monitor.evaluate(trusted_goal=trusted_goal, current_intent=execution.current_intent)
        action = self.action_monitor.evaluate(trusted_goal=trusted_goal, tool_call=execution.tool_call)
        fused_risk, confidence, weights = fuse_risk(reasoning=reasoning, intent=intent, action=action, config=self.config)
        ema_risk = self.ema_tracker.update(session_id, fused_risk)
        level = _decision_level(ema_risk, self.config)
        return SentinelDecision(
            session_id=session_id,
            reasoning_similarity=reasoning.similarity,
            intent_similarity=intent.similarity,
            action_similarity=action.similarity,
            reasoning_drift=reasoning.drift,
            intent_drift=intent.drift,
            action_drift=action.drift,
            fused_risk=fused_risk,
            ema_risk=ema_risk,
            confidence=confidence,
            decision=level,
            reason=_reason(level, confidence),
            metadata={
                "weights": weights,
                "available_monitors": [
                    name
                    for name, result in (("reasoning", reasoning), ("intent", intent), ("action", action))
                    if result.available
                ],
                "embedding_model": self.embedder.model_name,
            },
        )


def _decision_level(risk: float, config: SentinelConfig) -> SentinelDecisionLevel:
    if risk < config.allow_threshold:
        return SentinelDecisionLevel.ALLOW
    if risk < config.observe_threshold:
        return SentinelDecisionLevel.OBSERVE
    if risk < config.review_threshold:
        return SentinelDecisionLevel.REVIEW
    return SentinelDecisionLevel.BLOCK


def _reason(level: SentinelDecisionLevel, confidence: float) -> str:
    return f"Sentinel produced {level.value} drift signal with confidence {confidence:.2f}."
