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
from aegisvault.layer0 import Layer0Decision, Layer0Validator
from aegisvault.sentinel import SentinelExecutionState, SentinelMonitor, ToolCallState
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
    "Layer0Decision",
    "Layer0Validator",
    "MalformedEvaluatorResponseError",
    "PolicyLoadError",
    "PolicyValidationError",
    "SentinelExecutionState",
    "SentinelMonitor",
    "TerminatedBy",
    "ToolCallState",
    "UnsupportedCallableError",
    "Verdict",
]
