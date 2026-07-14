"""Shared Sentinel monitor utilities."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from aegisvault.runtime.action_gate.cosine import cosine_similarity
from aegisvault.runtime.goal_vault.embedding import GoalEmbedder


def normalize_text(text: str) -> str:
    """Normalize monitor text deterministically."""

    return re.sub(r"\s+", " ", text.strip().casefold())


def similarity_to_drift(similarity: float) -> float:
    """Convert cosine similarity into a bounded drift score."""

    return max(0.0, min(1.0, 1.0 - max(0.0, similarity)))


def embed_similarity(embedder: GoalEmbedder, left: str, right: str) -> float:
    """Embed two strings and compute cosine similarity."""

    return cosine_similarity(embedder.embed(left), embedder.embed(right))


def safe_action_text(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Normalize a tool call into deterministic action text."""

    safe_args = json.dumps(_json_safe(arguments), ensure_ascii=False, sort_keys=True)
    return f"{tool_name.strip()}({safe_args})"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(raw_value) for key, raw_value in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value
