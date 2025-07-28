"""
Tests for UnifiedChatMessageHistory compatibility adapter.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from unified_checkpointer.checkpointer import UnifiedCheckpointer
from unified_checkpointer.compatibility import UnifiedChatMessageHistory


class TestUnifiedChatMessageHistory:
    """Test UnifiedChatMessageHistory compatibility with BaseChatMessageHistory."""

    @pytest.fixture
    def mock_checkpointer(self):
        """Create a mock UnifiedCheckpointer."""
        return Mock(spec=UnifiedCheckpointer)

    @pytest.fixture
    def chat_history(self, mock_checkpointer):
        """Create UnifiedChatMessageHistory with mocked dependencies."""
        with patch("unified_checkpointer.compatibility.chat_message_history.UnifiedCheckpointer", return_value=mock_checkpointer):
            return UnifiedChatMessageHistory(
                session_id="test-session",
                qdrant_url="http://localhost:6333",
            )

    def test_initialization(self, chat_history) -> None:
        """Test proper initialization."""
        assert chat_history.session_id == "test-session"
        assert chat_history._thread_id == "test-session"
        assert chat_history._checkpoint_id is not None

    def test_add_user_message(self, chat_history, mock_checkpointer) -> None:
        """Test adding user messages."""
        # Setup mock to return empty history initially
        mock_checkpointer.get_tuple.return_value = None

        # Add user message
        chat_history.add_user_message("Hello!")

        # Verify put was called
        mock_checkpointer.put.assert_called_once()
        call_args = mock_checkpointer.put.call_args

        # Check checkpoint structure
        checkpoint = call_args[0][1]
        assert "channel_values" in checkpoint
        assert "messages" in checkpoint["channel_values"]

        # Verify message was added
        messages = checkpoint["channel_values"]["messages"]
        assert len(messages) == 1
        assert messages[0]["type"] == "human"
        assert messages[0]["data"]["content"] == "Hello!"

    def test_add_ai_message(self, chat_history, mock_checkpointer) -> None:
        """Test adding AI messages."""
        mock_checkpointer.get_tuple.return_value = None

        chat_history.add_ai_message("Hi there!")

        mock_checkpointer.put.assert_called_once()
        checkpoint = mock_checkpointer.put.call_args[0][1]

        messages = checkpoint["channel_values"]["messages"]
        assert len(messages) == 1
        assert messages[0]["type"] == "ai"
        assert messages[0]["data"]["content"] == "Hi there!"

    def test_messages_property_empty(self, chat_history, mock_checkpointer) -> None:
        """Test messages property when history is empty."""
        mock_checkpointer.get_tuple.return_value = None

        messages = chat_history.messages
        assert messages == []

    def test_messages_property_with_history(self, chat_history, mock_checkpointer) -> None:
        """Test messages property with existing history."""
        # Create mock checkpoint with messages
        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint = {
            "channel_values": {
                "messages": [
                    {"type": "human", "data": {"content": "Hello"}},
                    {"type": "ai", "data": {"content": "Hi there"}},
                ],
            },
        }

        mock_checkpointer.get_tuple.return_value = mock_checkpoint

        messages = chat_history.messages

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there"

    def test_clear(self, chat_history, mock_checkpointer) -> None:
        """Test clearing history."""
        chat_history.clear()

        # Verify put was called with empty messages
        mock_checkpointer.put.assert_called_once()
        checkpoint = mock_checkpointer.put.call_args[0][1]

        assert checkpoint["channel_values"]["messages"] == []

    def test_multiple_messages(self, chat_history, mock_checkpointer) -> None:
        """Test adding multiple messages in sequence."""
        # Start with empty history
        mock_checkpointer.get_tuple.return_value = None

        # Add first message
        chat_history.add_user_message("First message")

        # Update mock to return the first message
        first_checkpoint = MagicMock()
        first_checkpoint.checkpoint = {
            "channel_values": {
                "messages": [
                    {"type": "human", "data": {"content": "First message"}},
                ],
            },
        }
        mock_checkpointer.get_tuple.return_value = first_checkpoint

        # Add second message
        chat_history.add_ai_message("Response")

        # Check second put call
        assert mock_checkpointer.put.call_count == 2
        second_checkpoint = mock_checkpointer.put.call_args[0][1]

        messages = second_checkpoint["channel_values"]["messages"]
        assert len(messages) == 2
        assert messages[0]["data"]["content"] == "First message"
        assert messages[1]["data"]["content"] == "Response"

    def test_error_handling(self, chat_history, mock_checkpointer) -> None:
        """Test error handling in messages property."""
        # Make get_tuple raise an exception
        mock_checkpointer.get_tuple.side_effect = Exception("Connection error")

        # Should return empty list on error
        messages = chat_history.messages
        assert messages == []

    def test_repr(self, chat_history, mock_checkpointer) -> None:
        """Test string representation."""
        mock_checkpointer.get_tuple.return_value = None

        repr_str = repr(chat_history)
        assert "UnifiedChatMessageHistory" in repr_str
        assert "session_id='test-session'" in repr_str
        assert "message_count=0" in repr_str


class TestIntegrationScenarios:
    """Test integration scenarios with LangChain patterns."""

    @pytest.fixture
    def mock_setup(self):
        """Setup mocks for integration tests."""
        mock_checkpointer = Mock(spec=UnifiedCheckpointer)

        with patch("unified_checkpointer.compatibility.chat_message_history.UnifiedCheckpointer", return_value=mock_checkpointer):
            history = UnifiedChatMessageHistory(
                session_id="integration-test",
                qdrant_url="http://localhost:6333",
            )
            return history, mock_checkpointer

    def test_conversation_flow(self, mock_setup) -> None:
        """Test typical conversation flow."""
        history, mock_checkpointer = mock_setup

        # Start with empty history
        mock_checkpointer.get_tuple.return_value = None

        # Simulate conversation
        test_conversation = [
            ("human", "What's the weather like?"),
            ("ai", "I don't have access to real-time weather data."),
            ("human", "Can you help me write code?"),
            ("ai", "Of course! I'd be happy to help with coding."),
        ]

        # Track checkpoint updates
        checkpoints = []

        def capture_checkpoint(config, checkpoint, metadata, new_versions) -> None:
            checkpoints.append(checkpoint)

        mock_checkpointer.put.side_effect = capture_checkpoint

        # Add messages
        for role, content in test_conversation[:2]:
            if role == "human":
                history.add_user_message(content)
            else:
                history.add_ai_message(content)

        # Verify checkpoints were created
        assert len(checkpoints) == 2

        # Check final checkpoint
        final_messages = checkpoints[-1]["channel_values"]["messages"]
        assert len(final_messages) == 2
        assert final_messages[0]["data"]["content"] == test_conversation[0][1]
        assert final_messages[1]["data"]["content"] == test_conversation[1][1]
