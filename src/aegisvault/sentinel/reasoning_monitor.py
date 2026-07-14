"""Sentinel reasoning monitor."""

from __future__ import annotations

from aegisvault.runtime.goal_vault.embedding import GoalEmbedder
from aegisvault.sentinel.common import embed_similarity, normalize_text, similarity_to_drift
from aegisvault.sentinel.models import MonitorResult


class ReasoningMonitor:
    """Compare trusted goal against available Qwen reasoning."""

    def __init__(self, embedder: GoalEmbedder) -> None:
        self.embedder = embedder

    def evaluate(self, *, trusted_goal: str, reasoning: str | None) -> MonitorResult:
        """Return reasoning drift, or unavailable when reasoning is absent."""

        if not reasoning or not reasoning.strip():
            return MonitorResult(
                similarity=None,
                drift=None,
                available=False,
                reason="Reasoning unavailable.",
            )
        similarity = embed_similarity(self.embedder, normalize_text(trusted_goal), normalize_text(reasoning))
        return MonitorResult(
            similarity=similarity,
            drift=similarity_to_drift(similarity),
            available=True,
            reason="Reasoning compared against trusted goal.",
        )
