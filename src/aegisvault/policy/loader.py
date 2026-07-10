"""YAML policy loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aegisvault.exceptions import PolicyLoadError, PolicyValidationError
from aegisvault.policy.models import DomainPolicy


def load_policy(path: str | Path) -> DomainPolicy:
    """Load and validate a domain policy from a YAML file."""

    policy_path = Path(path)
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(f"Unable to read policy file {policy_path}: {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PolicyLoadError(f"Invalid YAML in policy file {policy_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise PolicyValidationError("Invalid AegisVault policy: top-level YAML document must be a mapping")

    try:
        return DomainPolicy.model_validate(data)
    except ValidationError as exc:
        raise PolicyValidationError(f"Invalid AegisVault policy {policy_path}: {exc}") from exc
