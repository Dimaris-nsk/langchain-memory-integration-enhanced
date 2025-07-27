"""
UnifiedChatMessageHistory - Adapter for LangChain BaseChatMessageHistory.

This module provides a bridge between UnifiedCheckpointer and the legacy
LangChain BaseChatMessageHistory interface for backward compatibility.
"""

from uuid import uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

from unified_checkpointer.checkpointer import UnifiedCheckpointer
from unified_checkpointer.config import UnifiedCheckpointerConfig


class UnifiedChatMessageHistory(BaseChatMessageHistory):
    """
    Adapter class that implements BaseChatMessageHistory interface
    using UnifiedCheckpointer as the backend.

    This allows existing LangChain applications to use unified-memory
    without code changes.

    Example:
        ```python
        # Legacy code that works unchanged
        history = UnifiedChatMessageHistory(
            session_id="user-123",
            qdrant_url="http://localhost:6333"
        )

        history.add_user_message("Hello!")
        history.add_ai_message("Hi there!")

        messages = history.messages
        ```
    """

    def __init__(
        self,
        session_id: str,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "langchain_chat_history",
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize UnifiedChatMessageHistory.

        Args:
            session_id: Unique identifier for the conversation
            qdrant_url: Qdrant server URL
            collection_name: Collection name in Qdrant
            api_key: Optional API key for Qdrant
            **kwargs: Additional configuration options
        """
        self.session_id = session_id

        # Create config for UnifiedCheckpointer
        config = UnifiedCheckpointerConfig(
            unified_memory_url=qdrant_url,
            collection_name=collection_name,
            api_key=api_key,
            **kwargs,
        )

        # Initialize checkpointer
        self._checkpointer = UnifiedCheckpointer(config)

        # Thread ID is session_id for compatibility
        self._thread_id = session_id

        # Checkpoint ID for current session
        self._checkpoint_id = str(uuid4())

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Retrieve all messages from the conversation history.

        Returns:
            List of BaseMessage objects
        """
        try:
            # Get latest checkpoint
            checkpoint_tuple = self._checkpointer.get_tuple(
                {"configurable": {"thread_id": self._thread_id}},
            )

            if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
                return []

            # Extract messages from checkpoint
            checkpoint_data = checkpoint_tuple.checkpoint

            # Messages are stored in channel_values
            if "messages" in checkpoint_data.get("channel_values", {}):
                messages_data = checkpoint_data["channel_values"]["messages"]

                # Convert from dict format to BaseMessage objects
                if isinstance(messages_data, list):
                    return messages_from_dict(messages_data)

            return []

        except Exception:
            # Log error but return empty list for compatibility
            return []

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the history.

        Args:
            message: BaseMessage to add
        """
        # Get current messages
        current_messages = self.messages

        # Add new message
        current_messages.append(message)

        # Convert to dict format for storage
        messages_data = messages_to_dict(current_messages)

        # Create checkpoint with updated messages
        checkpoint = {
            "v": 1,
            "ts": str(uuid4()),  # Use UUID as timestamp
            "id": self._checkpoint_id,
            "channel_values": {"messages": messages_data},
            "channel_versions": {},
            "versions_seen": {},
        }

        # Save checkpoint
        config = {
            "configurable": {
                "thread_id": self._thread_id,
                "checkpoint_id": self._checkpoint_id,
            },
        }

        self._checkpointer.put(config, checkpoint, {}, {})

        # Generate new checkpoint ID for next save
        self._checkpoint_id = str(uuid4())

    def add_user_message(self, message: str | BaseMessage) -> None:
        """
        Add a user message to the history.

        Args:
            message: Message string or BaseMessage
        """
        if isinstance(message, str):
            message = HumanMessage(content=message)
        self.add_message(message)

    def add_ai_message(self, message: str | BaseMessage) -> None:
        """
        Add an AI message to the history.

        Args:
            message: Message string or BaseMessage
        """
        if isinstance(message, str):
            message = AIMessage(content=message)
        self.add_message(message)

    def clear(self) -> None:
        """Clear all messages from the history."""
        # Create empty checkpoint
        checkpoint = {
            "v": 1,
            "ts": str(uuid4()),
            "id": self._checkpoint_id,
            "channel_values": {"messages": []},
            "channel_versions": {},
            "versions_seen": {},
        }

        config = {
            "configurable": {
                "thread_id": self._thread_id,
                "checkpoint_id": self._checkpoint_id,
            },
        }

        self._checkpointer.put(config, checkpoint, {}, {})

        # Generate new checkpoint ID
        self._checkpoint_id = str(uuid4())

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnifiedChatMessageHistory("
            f"session_id='{self.session_id}', "
            f"message_count={len(self.messages)})"
        )
