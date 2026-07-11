from __future__ import annotations

import os

import pytest

from evaluation.scripts.validate_environment import main as validate_main


@pytest.mark.gemini
@pytest.mark.skipif(os.getenv("AEGISVAULT_RUN_GEMINI_TESTS") != "1", reason="Gemini/Ollama integration test disabled")
def test_stage2_environment_validation_integration() -> None:
    assert validate_main() == 0
