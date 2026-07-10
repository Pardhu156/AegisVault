from __future__ import annotations

import os

import pytest

from aegisvault.evaluators import OllamaScopeEvaluator
from aegisvault.types import GateType, Verdict


@pytest.mark.ollama
@pytest.mark.skipif(os.getenv("AEGISVAULT_RUN_OLLAMA_TESTS") != "1", reason="Ollama integration test disabled")
def test_ollama_evaluator_integration(policy) -> None:
    evaluator = OllamaScopeEvaluator.from_policy(policy)

    result = evaluator.evaluate("Where is my order?", policy, GateType.REQUEST)

    assert result.verdict in {Verdict.ALLOW, Verdict.BLOCK}
    assert result.confidence is not None
