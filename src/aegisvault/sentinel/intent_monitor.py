"""Sentinel intent monitor."""

from __future__ import annotations

from aegisvault.runtime.goal_vault.embedding import GoalEmbedder
from aegisvault.sentinel.common import embed_similarity, normalize_text, similarity_to_drift
from aegisvault.sentinel.models import MonitorResult


class IntentMonitor:
    """Compare trusted goal against current structured intent."""

    def __init__(self, embedder: GoalEmbedder) -> None:
        self.embedder = embedder

    def evaluate(self, *, trusted_goal: str, current_intent: str | None) -> MonitorResult:
        """Return current-intent drift."""

        if not current_intent or not current_intent.strip():
            return MonitorResult(
                similarity=None,
                drift=None,
                available=False,
                reason="Current intent unavailable.",
            )
        similarity = embed_similarity(self.embedder, normalize_text(trusted_goal), normalize_text(current_intent))
        return MonitorResult(
            similarity=similarity,
            drift=similarity_to_drift(similarity),
            available=True,
            reason="Current intent compared against trusted goal.",
        )
