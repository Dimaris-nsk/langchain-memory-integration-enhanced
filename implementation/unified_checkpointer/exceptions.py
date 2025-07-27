"""Custom exceptions for UnifiedCheckpointer."""


class UnifiedCheckpointerError(Exception):
    """Base exception for UnifiedCheckpointer."""


class ConfigurationError(UnifiedCheckpointerError):
    """Raised when configuration is invalid."""


class ConnectionError(UnifiedCheckpointerError):
    """Raised when connection to unified-memory fails."""


class SerializationError(UnifiedCheckpointerError):
    """Raised when serialization/deserialization fails."""


class CheckpointNotFoundError(UnifiedCheckpointerError):
    """Raised when requested checkpoint doesn't exist."""


class UnifiedMemoryUnavailable(UnifiedCheckpointerError):
    """Raised when unified-memory service is unavailable."""


class TransientError(UnifiedCheckpointerError):
    """Raised for transient errors that should be retried."""
