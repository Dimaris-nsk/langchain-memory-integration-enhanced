"""
Base adapter for providing LangChain Memory compatibility.
"""

from abc import ABC

from langchain_core.memory import BaseMemory

from unified_checkpointer.checkpointer import UnifiedCheckpointer
from unified_checkpointer.config import UnifiedCheckpointerConfig


class UnifiedMemoryAdapter(BaseMemory, ABC):
    """
    Base adapter that wraps UnifiedCheckpointer to provide
    backward compatibility with LangChain Memory interfaces.
    """

    def __init__(
        self,
        checkpointer: UnifiedCheckpointer | None = None,
        thread_id: str | None = None,
        memory_key: str = "history",
        input_key: str | None = None,
        output_key: str | None = None,
        return_messages: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the memory adapter.

        Args:
            checkpointer: UnifiedCheckpointer instance
            thread_id: Thread/conversation ID
            memory_key: Key to store memory under
            input_key: Key for input in load_memory_variables
            output_key: Key for output in save_context
            return_messages: Whether to return messages or string
        """
        super().__init__()

        # Initialize checkpointer if not provided
        if checkpointer is None:
            config = UnifiedCheckpointerConfig(**kwargs)
            checkpointer = UnifiedCheckpointer(config)

        self.checkpointer = checkpointer
        self.thread_id = thread_id or "default"
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables."""
        return [self.memory_key]

    def clear(self) -> None:
        """Clear memory contents."""
        # Get current checkpoint and clear channel values
        checkpoint_tuple = self.checkpointer.get_tuple(
            {"configurable": {"thread_id": self.thread_id}},
        )

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            # Create new checkpoint with empty channel values
            checkpoint = checkpoint_tuple.checkpoint
            checkpoint["channel_values"] = {}

            # Save cleared checkpoint
            self.checkpointer.put(
                {"configurable": {"thread_id": self.thread_id}},
                checkpoint,
                {},  # metadata
            )
