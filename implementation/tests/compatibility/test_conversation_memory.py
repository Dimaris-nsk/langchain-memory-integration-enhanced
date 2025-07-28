"""
Tests for UnifiedConversationBufferMemory and UnifiedConversationSummaryMemory.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from unified_checkpointer.compatibility import (
    UnifiedChatMessageHistory,
    UnifiedConversationBufferMemory,
    UnifiedConversationSummaryMemory,
)


class TestUnifiedConversationBufferMemory:
    """Test UnifiedConversationBufferMemory compatibility."""

    @pytest.fixture
    def mock_chat_history(self):
        """Create mock chat message history."""
        mock = Mock(spec=UnifiedChatMessageHistory)
        mock.messages = []
        return mock

    @pytest.fixture
    def buffer_memory(self, mock_chat_history):
        """Create UnifiedConversationBufferMemory with mocked history."""
        with patch("unified_checkpointer.compatibility.conversation_memory.UnifiedChatMessageHistory", return_value=mock_chat_history):
            memory = UnifiedConversationBufferMemory(
                session_id="test-session",
                qdrant_url="http://localhost:6333",
            )
            memory.chat_memory = mock_chat_history
            return memory

    def test_initialization(self, buffer_memory) -> None:
        """Test proper initialization."""
        assert buffer_memory.session_id == "test-session"
        assert buffer_memory.memory_key == "history"
        assert buffer_memory.human_prefix == "Human"
        assert buffer_memory.ai_prefix == "AI"

    def test_memory_variables(self, buffer_memory) -> None:
        """Test memory variables property."""
        assert buffer_memory.memory_variables == ["history"]

    def test_load_memory_variables_empty(self, buffer_memory, mock_chat_history) -> None:
        """Test loading memory when empty."""
        mock_chat_history.messages = []

        result = buffer_memory.load_memory_variables({})

        assert "history" in result
        assert result["history"] == ""  # Empty string when no messages

    def test_load_memory_variables_with_messages(self, buffer_memory, mock_chat_history) -> None:
        """Test loading memory with messages."""
        mock_chat_history.messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        result = buffer_memory.load_memory_variables({})

        assert "history" in result
        assert "Human: Hello" in result["history"]
        assert "AI: Hi there!" in result["history"]

    def test_save_context(self, buffer_memory, mock_chat_history) -> None:
        """Test saving conversation context."""
        inputs = {"input": "What's the weather?"}
        outputs = {"output": "I don't have weather data."}

        buffer_memory.save_context(inputs, outputs)

        # Verify messages were added
        assert mock_chat_history.add_user_message.called
        assert mock_chat_history.add_ai_message.called

        # Check message content
        mock_chat_history.add_user_message.assert_called_with("What's the weather?")
        mock_chat_history.add_ai_message.assert_called_with("I don't have weather data.")

    def test_clear(self, buffer_memory, mock_chat_history) -> None:
        """Test clearing memory."""
        buffer_memory.clear()
        mock_chat_history.clear.assert_called_once()

    def test_buffer_as_messages(self, buffer_memory, mock_chat_history) -> None:
        """Test buffer_as_messages property."""
        test_messages = [
            HumanMessage(content="Test"),
            AIMessage(content="Response"),
        ]
        mock_chat_history.messages = test_messages

        assert buffer_memory.buffer_as_messages == test_messages

    def test_buffer_as_str(self, buffer_memory, mock_chat_history) -> None:
        """Test buffer_as_str property."""
        mock_chat_history.messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]

        buffer_str = buffer_memory.buffer_as_str
        assert "Human: Hello" in buffer_str
        assert "AI: Hi!" in buffer_str


class TestUnifiedConversationSummaryMemory:
    """Test UnifiedConversationSummaryMemory compatibility."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for summarization."""
        mock = Mock()
        mock.predict.return_value = "Summary of conversation"
        return mock

    @pytest.fixture
    def mock_chat_history(self):
        """Create mock chat message history."""
        mock = Mock(spec=UnifiedChatMessageHistory)
        mock.messages = []
        return mock

    @pytest.fixture
    def summary_memory(self, mock_chat_history, mock_llm):
        """Create UnifiedConversationSummaryMemory."""
        with patch("unified_checkpointer.compatibility.conversation_memory.UnifiedChatMessageHistory", return_value=mock_chat_history):
            memory = UnifiedConversationSummaryMemory(
                session_id="test-session",
                llm=mock_llm,
                qdrant_url="http://localhost:6333",
            )
            memory.chat_memory = mock_chat_history
            return memory

    def test_initialization(self, summary_memory) -> None:
        """Test proper initialization."""
        assert summary_memory.session_id == "test-session"
        assert summary_memory.summary == ""
        assert summary_memory.buffer == ""
        assert summary_memory.summary_update_messages == 5

    def test_update_summary(self, summary_memory, mock_llm) -> None:
        """Test summary update logic."""
        # Set up test buffer
        summary_memory.buffer = "Human: Hello\nAI: Hi!\nHuman: How are you?\nAI: I'm doing well."

        # Call update_summary
        summary_memory._update_summary()

        # Verify LLM was called with correct prompt
        mock_llm.predict.assert_called()
        call_args = mock_llm.predict.call_args[0][0]
        assert "Progressively summarize" in call_args
        assert summary_memory.buffer in call_args

    def test_load_memory_variables_with_summary(self, summary_memory) -> None:
        """Test loading memory with existing summary."""
        summary_memory.summary = "Previous conversation about weather."
        summary_memory.buffer = "Human: What's new?\nAI: Nothing much."

        result = summary_memory.load_memory_variables({})

        assert "history" in result
        assert "Previous conversation about weather" in result["history"]
        assert "What's new?" in result["history"]
        assert "Nothing much" in result["history"]

    def test_save_context_with_summary_update(self, summary_memory, mock_chat_history, mock_llm) -> None:
        """Test saving context triggers summary update when threshold reached."""
        # Add messages to reach threshold
        for i in range(5):
            inputs = {"input": f"Question {i}"}
            outputs = {"output": f"Answer {i}"}
            summary_memory.save_context(inputs, outputs)

        # Verify summary was updated
        assert mock_llm.predict.called
        assert summary_memory.summary != ""

    def test_prune(self, summary_memory, mock_chat_history) -> None:
        """Test pruning conversation history."""
        # Set up summary and buffer
        summary_memory.summary = "Previous conversation summary."
        summary_memory.buffer = "Recent messages"

        # Call prune
        summary_memory.prune()

        # Verify chat history was cleared
        mock_chat_history.clear.assert_called_once()
        # Summary should remain, buffer should be cleared
        assert summary_memory.summary == "Previous conversation summary."
        assert summary_memory.buffer == ""

    def test_no_llm_fallback(self) -> None:
        """Test behavior when no LLM is provided."""
        with pytest.raises(ValueError, match="LLM is required"):
            UnifiedConversationSummaryMemory(
                session_id="test-session",
                llm=None,
                qdrant_url="http://localhost:6333",
            )

    def test_clear_with_summary(self, summary_memory, mock_chat_history) -> None:
        """Test clearing memory also clears summary."""
        summary_memory.summary = "Some summary"
        summary_memory.buffer = "Some buffer"

        summary_memory.clear()

        mock_chat_history.clear.assert_called_once()
        assert summary_memory.summary == ""
        assert summary_memory.buffer == ""

    def test_save_context_different_keys(self, summary_memory, mock_chat_history) -> None:
        """Test save_context with different input/output keys."""
        summary_memory.input_key = "question"
        summary_memory.output_key = "answer"

        inputs = {"question": "What's the time?", "other": "ignored"}
        outputs = {"answer": "It's 3 PM", "other": "also ignored"}

        summary_memory.save_context(inputs, outputs)

        mock_chat_history.add_user_message.assert_called_with("What's the time?")
        mock_chat_history.add_ai_message.assert_called_with("It's 3 PM")
