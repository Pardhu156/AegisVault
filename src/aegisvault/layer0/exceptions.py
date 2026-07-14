"""Layer 0 exceptions."""

from __future__ import annotations


class Layer0Error(Exception):
    """Base exception for Layer 0 validation errors."""


class Layer0ConfigurationError(Layer0Error):
    """Raised when Layer 0 configuration is invalid."""


class Layer0ValidationError(Layer0Error):
    """Raised for unexpected Layer 0 validation failures."""
