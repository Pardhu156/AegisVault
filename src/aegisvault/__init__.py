"""AegisVault public package API."""

from aegisvault.exceptions import (
    AegisVaultError,
    EvaluatorError,
    EvaluatorTimeoutError,
    MalformedEvaluatorResponseError,
    PolicyLoadError,
    PolicyValidationError,
    UnsupportedCallableError,
)
from aegisvault.guard import AegisVault
from aegisvault.types import EvaluationContext, GateDecision, GateType, GuardResult, TerminatedBy, Verdict

__all__ = [
    "AegisVault",
    "AegisVaultError",
    "EvaluationContext",
    "EvaluatorError",
    "EvaluatorTimeoutError",
    "GateDecision",
    "GateType",
    "GuardResult",
    "MalformedEvaluatorResponseError",
    "PolicyLoadError",
    "PolicyValidationError",
    "TerminatedBy",
    "UnsupportedCallableError",
    "Verdict",
]
