"""Custom exceptions raised by AegisVault."""


class AegisVaultError(Exception):
    """Base exception for all AegisVault errors."""


class PolicyLoadError(AegisVaultError):
    """Raised when a policy file cannot be read or parsed."""


class PolicyValidationError(AegisVaultError):
    """Raised when a policy is structurally invalid."""


class EvaluatorError(AegisVaultError):
    """Raised when a scope evaluator fails."""


class EvaluatorTimeoutError(EvaluatorError):
    """Raised when a scope evaluator times out."""


class MalformedEvaluatorResponseError(EvaluatorError):
    """Raised when an evaluator returns invalid or malformed data."""


class UnsupportedCallableError(AegisVaultError):
    """Raised when a callable cannot be wrapped by AegisVault."""
