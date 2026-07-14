"""Per-session Sentinel EMA drift tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EmaDriftTracker:
    """In-memory per-session EMA drift tracker."""

    alpha: float = 0.40
    _state: dict[str, float] = field(default_factory=dict)

    def update(self, session_id: str, fused_risk: float) -> float:
        """Update and return EMA risk for a session."""

        previous = self._state.get(session_id)
        if previous is None:
            value = fused_risk
        else:
            value = self.alpha * fused_risk + (1.0 - self.alpha) * previous
        self._state[session_id] = value
        return value

    def get(self, session_id: str) -> float | None:
        """Return current EMA state for a session."""

        return self._state.get(session_id)
