"""Pydantic policy models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegisvault.exceptions import PolicyValidationError


SUPPORTED_POLICY_VERSIONS = {"1.0"}


class LowConfidenceAction(str, Enum):
    """Actions available when evaluator confidence is below threshold."""

    ALLOW = "allow"
    BLOCK = "block"
    CLARIFY = "clarify"
    REPLACE = "replace"


class FallbackAction(str, Enum):
    """Runtime fallback actions for evaluator failures."""

    ALLOW = "allow"
    BLOCK = "block"
    CLARIFY = "clarify"
    REPLACE = "replace"


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_threshold: float = Field(ge=0.0, le=1.0)
    block_threshold: float = Field(ge=0.0, le=1.0)
    low_confidence_action: LowConfidenceAction


class GatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: GateConfig
    response: GateConfig

    @model_validator(mode="after")
    def validate_gate_actions(self) -> "GatesConfig":
        if self.request.low_confidence_action == LowConfidenceAction.REPLACE:
            raise ValueError("gates.request.low_confidence_action cannot be 'replace'")
        if self.response.low_confidence_action == LowConfidenceAction.CLARIFY:
            raise ValueError("gates.response.low_confidence_action cannot be 'clarify'")
        return self


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama"]
    model: str = Field(min_length=1)
    base_url: str = Field(default="http://localhost:11434", min_length=1)
    timeout_seconds: float = Field(default=30, gt=0)
    temperature: float = Field(default=0, ge=0.0)


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_failure_action: FallbackAction = FallbackAction.BLOCK
    malformed_output_action: FallbackAction = FallbackAction.BLOCK


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    output_path: str = "logs/aegisvault.jsonl"
    include_request_text: bool = True
    include_response_text: bool = True


class DeterministicChecksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_request_chars: int = Field(default=8000, gt=0)
    max_response_chars: int = Field(default=12000, gt=0)
    blocked_phrases: list[str] = Field(default_factory=list)
    blocked_keywords: list[str] = Field(default_factory=list)
    keyword_case_insensitive: bool = True


class MessagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_blocked: str = "I can only help with requests that fit this application's purpose."
    request_clarify: str = "Please restate your request within this application's purpose."
    response_blocked: str = "I cannot provide that response because it falls outside this application's purpose."
    response_replaced: str = "I cannot provide that response because it falls outside this application's purpose."


class DomainPolicy(BaseModel):
    """Complete domain policy loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    version: str
    application: ApplicationConfig
    purpose: str = Field(min_length=1)
    allowed_topics: list[str] = Field(min_length=1)
    blocked_topics: list[str] = Field(default_factory=list)
    gates: GatesConfig
    evaluator: EvaluatorConfig
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    checks: DeterministicChecksConfig = Field(default_factory=DeterministicChecksConfig)
    messages: MessagesConfig = Field(default_factory=MessagesConfig)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value not in SUPPORTED_POLICY_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_POLICY_VERSIONS))
            raise ValueError(f"unsupported policy version {value!r}; supported versions: {supported}")
        return value

    @field_validator("allowed_topics", "blocked_topics")
    @classmethod
    def validate_topic_strings(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("topics must be non-empty strings")
        return value


def validation_error_from_exception(exc: Exception) -> PolicyValidationError:
    """Convert a Pydantic error into AegisVault's public validation exception."""

    return PolicyValidationError(f"Invalid AegisVault policy: {exc}")
