"""
Conversation Memory Adapters for LangChain compatibility.

This module provides UnifiedConversationBufferMemory and
UnifiedConversationSummaryMemory classes that are drop-in replacements
for the legacy LangChain memory classes.
"""

from typing import Any

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string

from .chat_message_history import UnifiedChatMessageHistory


class UnifiedConversationBufferMemory(BaseChatMemory):
    """
    Drop-in replacement for ConversationBufferMemory using UnifiedCheckpointer.

    Example:
        ```python
        # Works exactly like original ConversationBufferMemory
        memory = UnifiedConversationBufferMemory(
            session_id="user-123",
            qdrant_url="http://localhost:6333"
        )

        # In a chain
        chain = LLMChain(llm=llm, memory=memory, prompt=prompt)
        chain.run("Hello!")
        ```
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    # UnifiedCheckpointer specific
    session_id: str
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "langchain_memory"
    api_key: str | None = None

    @property
    def buffer(self) -> str | list[BaseMessage]:
        """Return the buffer content."""
        if self.return_messages:
            return self.chat_memory.messages
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Return buffer as list of messages."""
        return self.chat_memory.messages

    @property
    def buffer_as_str(self) -> str:
        """Return buffer as string."""
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables."""
        return [self.memory_key]

    def __init__(self, **kwargs) -> None:
        """Initialize UnifiedConversationBufferMemory."""
        # Extract unified-specific params
        session_id = kwargs.pop("session_id", None)
        if not session_id:
            msg = "session_id is required for UnifiedConversationBufferMemory"
            raise ValueError(
                msg,
            )

        qdrant_url = kwargs.pop("qdrant_url", "http://localhost:6333")
        collection_name = kwargs.pop("collection_name", "langchain_memory")
        api_key = kwargs.pop("api_key", None)

        # Create chat message history if not provided
        if "chat_memory" not in kwargs:
            kwargs["chat_memory"] = UnifiedChatMessageHistory(
                session_id=session_id,
                qdrant_url=qdrant_url,
                collection_name=collection_name,
                api_key=api_key,
            )

        # Set instance attributes
        self.session_id = session_id
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.api_key = api_key

        # Initialize parent
        super().__init__(**kwargs)

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str = get_prompt_input_key(inputs, self.input_key)
        output_str = get_prompt_input_key(outputs, self.output_key)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()


class UnifiedConversationSummaryMemory(BaseChatMemory):
    """
    Drop-in replacement for ConversationSummaryMemory using UnifiedCheckpointer.

    This implementation stores both the full history and a summary.
    The summary is updated periodically based on the conversation length.

    Example:
        ```python
        memory = UnifiedConversationSummaryMemory(
            llm=llm,
            session_id="user-123",
            qdrant_url="http://localhost:6333"
        )

        chain = LLMChain(llm=llm, memory=memory, prompt=prompt)
        ```
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    # Summary specific
    summary: str = ""
    summary_message_cls: type = SystemMessage
    buffer: str = ""

    # UnifiedCheckpointer specific
    session_id: str
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "langchain_memory"
    api_key: str | None = None

    # Summary update frequency
    summary_update_messages: int = 5  # Update summary every N messages

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables."""
        return [self.memory_key]

    def __init__(self, **kwargs) -> None:
        """Initialize UnifiedConversationSummaryMemory."""
        # Extract unified-specific params
        session_id = kwargs.pop("session_id", None)
        if not session_id:
            msg = "session_id is required for UnifiedConversationSummaryMemory"
            raise ValueError(
                msg,
            )

        qdrant_url = kwargs.pop("qdrant_url", "http://localhost:6333")
        collection_name = kwargs.pop("collection_name", "langchain_memory")
        api_key = kwargs.pop("api_key", None)

        # Create chat message history if not provided
        if "chat_memory" not in kwargs:
            kwargs["chat_memory"] = UnifiedChatMessageHistory(
                session_id=session_id,
                qdrant_url=qdrant_url,
                collection_name=collection_name,
                api_key=api_key,
            )

        # Set instance attributes
        self.session_id = session_id
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.api_key = api_key

        # Initialize parent
        super().__init__(**kwargs)

        # Load existing summary if available
        self._load_summary()

    def _load_summary(self) -> None:
        """Load existing summary from checkpoint."""
        # This would load from checkpoint metadata
        # For now, we'll generate from messages if needed
        if len(self.chat_memory.messages) > 0 and not self.summary:
            self._update_summary()

    def _update_summary(self) -> None:
        """Update the summary using the LLM."""
        if not hasattr(self, "llm") or self.llm is None:
            # No LLM provided, use simple concatenation
            self.summary = get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
            return

        # Use LLM to generate summary
        # This is a simplified version - real implementation would use
        # the summarization chain from the original
        messages_str = get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        prompt = f"""Progressively summarize the lines of conversation provided,
adding onto the previous summary returning a new summary.

Current summary:
{self.summary}

New lines of conversation:
{messages_str}

New summary:"""

        try:
            self.summary = self.llm.predict(prompt)
        except Exception:
            # Fallback to simple concatenation
            self.summary = messages_str

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        # Include both summary and recent buffer
        if self.return_messages:
            # Return as messages
            summary_message = self.summary_message_cls(content=self.summary)
            return {self.memory_key: [summary_message, *self.chat_memory.messages[-4:]]}
        # Return as string
        if self.summary:
            return {self.memory_key: self.summary + "\n" + self.buffer}
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)

        # Update buffer
        self.buffer = get_buffer_string(
            self.chat_memory.messages[-2:],  # Last exchange
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        # Check if we should update summary
        if len(self.chat_memory.messages) % self.summary_update_messages == 0:
            self._update_summary()

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.summary = ""
        self.buffer = ""

    def prune(self) -> None:
        """Prune buffer if it gets too long."""
        # Keep summary but clear old messages
        if len(self.chat_memory.messages) > 10:
            # Update summary with all messages
            self._update_summary()
            # Keep only recent messages
            recent_messages = self.chat_memory.messages[-4:]
            self.chat_memory.clear()
            for msg in recent_messages:
                self.chat_memory.add_message(msg)