"""
Integration tests for the compatibility layer.
"""

from unittest.mock import Mock, patch

import pytest
from unified_checkpointer import UnifiedCheckpointer
from unified_checkpointer.compatibility import (
    UnifiedChatMessageHistory,
    UnifiedConversationBufferMemory,
)


class TestCompatibilityLayerIntegration:
    """Test integration between compatibility layer and UnifiedCheckpointer."""

    @pytest.fixture
    def mock_checkpointer(self):
        """Create mock UnifiedCheckpointer."""
        return Mock(spec=UnifiedCheckpointer)

    def test_chat_history_with_checkpointer(self, mock_checkpointer) -> None:
        """Test UnifiedChatMessageHistory works with UnifiedCheckpointer."""
        with patch("unified_checkpointer.compatibility.chat_message_history.UnifiedCheckpointer", return_value=mock_checkpointer):
            history = UnifiedChatMessageHistory(
                session_id="test-session",
                qdrant_url="http://localhost:6333",
            )

            # Test message operations
            history.add_user_message("Hello")
            history.add_ai_message("Hi there!")

            # Verify checkpointer was called correctly
            assert mock_checkpointer.put.called

    def test_memory_migration_scenario(self, mock_checkpointer) -> None:
        """Test migrating from legacy memory to unified checkpointer."""
        with patch("unified_checkpointer.compatibility.chat_message_history.UnifiedCheckpointer", return_value=mock_checkpointer):
            # Create legacy-style memory
            memory = UnifiedConversationBufferMemory(
                session_id="legacy-session",
                qdrant_url="http://localhost:6333",
            )

            # Use it like legacy memory
            memory.save_context(
                {"input": "What's your name?"},
                {"output": "I'm Claude."},
            )

            # Verify data was stored via checkpointer
            assert mock_checkpointer.put.called

    def test_thread_isolation(self, mock_checkpointer) -> None:
        """Test that different sessions are isolated."""
        mock_checkpointer.get_tuple.return_value = None

        with patch("unified_checkpointer.compatibility.chat_message_history.UnifiedCheckpointer", return_value=mock_checkpointer):
            # Create two separate histories
            history1 = UnifiedChatMessageHistory(session_id="session-1")
            history2 = UnifiedChatMessageHistory(session_id="session-2")

            # Add messages to each
            history1.add_user_message("Message for session 1")
            history2.add_user_message("Message for session 2")

            # Verify different thread_ids were used
            calls = mock_checkpointer.put.call_args_list
            thread_ids = [call[1]["config"]["configurable"]["thread_id"] for call in calls]
            assert "session-1" in thread_ids[0]
            assert "session-2" in thread_ids[1]
