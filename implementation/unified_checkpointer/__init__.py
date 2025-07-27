"""UnifiedCheckpointer - LangGraph checkpointer with unified-memory backend.

This package provides a BaseCheckpointSaver implementation that uses
unified-memory as its storage backend, adding enhanced features like
semantic search, tagging, and analytics.
"""

from .checkpointer import UnifiedCheckpointer
from .config import UnifiedCheckpointerConfig
from .exceptions import UnifiedCheckpointerError

__version__ = "0.1.0"
__all__ = [
    "UnifiedCheckpointer",
    "UnifiedCheckpointerConfig",
    "UnifiedCheckpointerError",
]
