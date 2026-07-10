from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aegisvault.exceptions import PolicyLoadError, PolicyValidationError
from aegisvault.policy import load_policy, validate_policy


def test_valid_policy_loading(policy_dict: dict, tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy_dict), encoding="utf-8")

    policy = load_policy(path)

    assert policy.application.name == "test-app"
    assert policy.evaluator.provider == "ollama"


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("version: [", encoding="utf-8")

    with pytest.raises(PolicyLoadError):
        load_policy(path)


def test_missing_required_policy_fields(policy_dict: dict) -> None:
    del policy_dict["purpose"]

    with pytest.raises(PolicyValidationError, match="purpose"):
        validate_policy(policy_dict)


def test_invalid_thresholds(policy_dict: dict) -> None:
    policy_dict["gates"]["request"]["allow_threshold"] = 1.2

    with pytest.raises(PolicyValidationError, match="allow_threshold"):
        validate_policy(policy_dict)


def test_unsupported_policy_version(policy_dict: dict) -> None:
    policy_dict["version"] = "9.9"

    with pytest.raises(PolicyValidationError, match="unsupported policy version"):
        validate_policy(policy_dict)
