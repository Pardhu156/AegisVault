"""Scope evaluator implementations."""

from aegisvault.evaluators.base import ScopeEvaluator
from aegisvault.evaluators.ollama import OllamaScopeEvaluator

__all__ = ["OllamaScopeEvaluator", "ScopeEvaluator"]
