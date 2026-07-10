from __future__ import annotations

from aegisvault.checks import check_request_text, check_response_text
from aegisvault.types import Verdict


def test_empty_input_handling(policy) -> None:
    result = check_request_text("   ", policy)

    assert result is not None
    assert result.verdict == Verdict.BLOCK
    assert result.confidence is None


def test_oversized_input_handling(policy) -> None:
    result = check_request_text("x" * 101, policy)

    assert result is not None
    assert result.verdict == Verdict.BLOCK


def test_empty_generated_response(policy) -> None:
    result = check_response_text("", policy)

    assert result is not None
    assert result.verdict == Verdict.BLOCK


def test_exact_blocked_phrase_matching(policy) -> None:
    result = check_request_text("This has a FORBIDDEN PHRASE in it.", policy)

    assert result is not None
    assert result.verdict == Verdict.BLOCK


def test_case_insensitive_keyword_matching(policy) -> None:
    result = check_request_text("This contains BLOCKEDWORD.", policy)

    assert result is not None
    assert result.verdict == Verdict.BLOCK
