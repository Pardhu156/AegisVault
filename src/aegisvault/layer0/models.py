"""Typed models for deterministic Layer 0 validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class Layer0Checkpoint(str, Enum):
    """Layer 0 validation checkpoints."""

    REQUEST = "request"
    TOOL_CALL = "tool_call"


class Layer0Action(str, Enum):
    """Rule and final decision actions."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class Layer0RiskLevel(str, Enum):
    """Coarse deterministic risk levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Layer0RuleResult:
    """Result produced by one deterministic Layer 0 rule."""

    rule_id: str
    matched: bool
    action: Layer0Action
    risk_level: Layer0RiskLevel
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Layer0Decision:
    """Aggregated Layer 0 decision."""

    allowed: bool
    decision: Layer0Action
    risk_level: Layer0RiskLevel
    checkpoint: Layer0Checkpoint
    reason: str
    matched_rules: tuple[Layer0RuleResult, ...] = ()
    rule_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_rules", tuple(self.matched_rules))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Layer0RequestInput:
    """Request validation input."""

    session_id: str | None
    request_text: Any
    domain: str | None = None
    policy_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    requested_goal_update: Any = None
    trusted_goal_exists: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Layer0ToolCallInput:
    """Tool-call validation input."""

    session_id: str | None
    tool_name: str | None
    arguments: Any
    domain: str | None = None
    policy_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tool_catalog: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "tool_catalog", _freeze_mapping(self.tool_catalog))


def thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable JSON-like mapping."""

    return {str(key): _thaw(raw_value) for key, raw_value in dict(value).items()}


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(raw_value) for key, raw_value in dict(value).items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
