"""Sentinel action monitor."""

from __future__ import annotations

from aegisvault.runtime.goal_vault.embedding import GoalEmbedder
from aegisvault.sentinel.common import embed_similarity, normalize_text, safe_action_text, similarity_to_drift
from aegisvault.sentinel.models import MonitorResult, ToolCallState


class ActionMonitor:
    """Compare trusted goal against a structured proposed tool call."""

    def __init__(self, embedder: GoalEmbedder) -> None:
        self.embedder = embedder

    def normalize_action(self, tool_call: ToolCallState | None) -> str | None:
        """Normalize a tool call into one textual action."""

        if tool_call is None or not tool_call.name.strip():
            return None
        return safe_action_text(tool_call.name, tool_call.arguments)

    def evaluate(self, *, trusted_goal: str, tool_call: ToolCallState | None) -> MonitorResult:
        """Return tool-call drift."""

        action_text = self.normalize_action(tool_call)
        if action_text is None:
            return MonitorResult(
                similarity=None,
                drift=None,
                available=False,
                reason="Tool call unavailable.",
            )
        similarity = embed_similarity(self.embedder, normalize_text(trusted_goal), normalize_text(action_text))
        return MonitorResult(
            similarity=similarity,
            drift=similarity_to_drift(similarity),
            available=True,
            reason="Tool call compared against trusted goal.",
            metadata={"action_text": action_text},
        )
