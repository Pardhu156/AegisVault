"""Lightweight deterministic checks that run before evaluator calls."""

from __future__ import annotations

from aegisvault.policy.models import DeterministicChecksConfig, DomainPolicy
from aegisvault.types import GateDecision, GateType, Verdict


def check_request_text(text: str, policy: DomainPolicy) -> GateDecision | None:
    """Return a deterministic request decision, or None when evaluation should continue."""

    return _check_text(text, policy.checks, GateType.REQUEST, policy.checks.max_request_chars, "request")


def check_response_text(text: str, policy: DomainPolicy) -> GateDecision | None:
    """Return a deterministic response decision, or None when evaluation should continue."""

    return _check_text(text, policy.checks, GateType.RESPONSE, policy.checks.max_response_chars, "response")


def _check_text(
    text: str,
    config: DeterministicChecksConfig,
    gate_type: GateType,
    max_chars: int,
    label: str,
) -> GateDecision | None:
    if not text.strip():
        return _decision(gate_type, f"Empty {label} text is not allowed.")

    if len(text) > max_chars:
        return _decision(gate_type, f"{label.capitalize()} text exceeds maximum length of {max_chars} characters.")

    matched_phrase = _match_blocked_phrase(text, config)
    if matched_phrase is not None:
        return _decision(gate_type, f"Text matched blocked phrase: {matched_phrase!r}.", {"matched": matched_phrase})

    matched_keyword = _match_blocked_keyword(text, config)
    if matched_keyword is not None:
        return _decision(gate_type, f"Text matched blocked keyword: {matched_keyword!r}.", {"matched": matched_keyword})

    return None


def _match_blocked_phrase(text: str, config: DeterministicChecksConfig) -> str | None:
    for phrase in config.blocked_phrases:
        if phrase and phrase in text:
            return phrase
    return None


def _match_blocked_keyword(text: str, config: DeterministicChecksConfig) -> str | None:
    haystack = text.lower() if config.keyword_case_insensitive else text
    for keyword in config.blocked_keywords:
        needle = keyword.lower() if config.keyword_case_insensitive else keyword
        if needle and needle in haystack:
            return keyword
    return None


def _decision(gate_type: GateType, reason: str, metadata: dict[str, str] | None = None) -> GateDecision:
    return GateDecision(
        verdict=Verdict.BLOCK,
        confidence=None,
        reason=reason,
        gate=gate_type,
        evaluator="deterministic",
        latency_ms=0.0,
        metadata=metadata or {},
    )
