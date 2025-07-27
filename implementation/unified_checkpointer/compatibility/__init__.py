"""
LangChain Compatibility Layer.

This package provides drop-in replacements for legacy LangChain memory classes,
allowing existing applications to use UnifiedCheckpointer without code changes.

Classes:
    - UnifiedChatMessageHistory: Replacement for BaseChatMessageHistory
    - UnifiedConversationBufferMemory: Replacement for ConversationBufferMemory
    - UnifiedConversationSummaryMemory: Replacement for ConversationSummaryMemory

Example:
    ```python
    # Replace this:
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory()

    # With this:
    from unified_checkpointer.compatibility import UnifiedConversationBufferMemory
    memory = UnifiedConversationBufferMemory(
        session_id="user-123",
        qdrant_url="http://localhost:6333"
    )
    ```
"""

from .chat_message_history import UnifiedChatMessageHistory
from .conversation_memory import (
    UnifiedConversationBufferMemory,
    UnifiedConversationSummaryMemory,
)

__all__ = [
    "UnifiedChatMessageHistory",
    "UnifiedConversationBufferMemory",
    "UnifiedConversationSummaryMemory",
]

# Version info
__version__ = "0.1.0"
