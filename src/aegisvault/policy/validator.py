"""Policy validation helpers."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from aegisvault.exceptions import PolicyValidationError
from aegisvault.policy.models import DomainPolicy


def validate_policy(data: dict[str, Any]) -> DomainPolicy:
    """Validate a raw dictionary as an AegisVault domain policy."""

    try:
        return DomainPolicy.model_validate(data)
    except ValidationError as exc:
        raise PolicyValidationError(f"Invalid AegisVault policy: {exc}") from exc
