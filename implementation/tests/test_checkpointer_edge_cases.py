"""Edge case and error handling tests for UnifiedCheckpointer."""

import uuid
from datetime import datetime

import pytest
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
)
from unified_checkpointer import UnifiedCheckpointer, UnifiedCheckpointerConfig
from unified_checkpointer.exceptions import (
    ConfigurationError,
)


class TestUnifiedCheckpointerEdgeCases:
    """Test edge cases and error handling."""

    def test_configuration_validation(self) -> None:
        """Test configuration validation raises proper errors."""
        # Test empty collection name
        with pytest.raises(ConfigurationError, match="collection_name cannot be empty"):
            config = UnifiedCheckpointerConfig(collection_name="")
            UnifiedCheckpointer(config)

        # Test negative cache size
        with pytest.raises(ConfigurationError, match="cache_size must be non-negative"):
            config = UnifiedCheckpointerConfig(cache_size=-1)
            UnifiedCheckpointer(config)

        # Test invalid connection pool size
        with pytest.raises(ConfigurationError, match="connection_pool_size must be at least 1"):
            config = UnifiedCheckpointerConfig(connection_pool_size=0)
            UnifiedCheckpointer(config)

        # Test invalid batch size
        with pytest.raises(ConfigurationError, match="batch_size must be at least 1"):
            config = UnifiedCheckpointerConfig(batch_size=0)
            UnifiedCheckpointer(config)

        # Test invalid retry attempts
        with pytest.raises(ConfigurationError, match="retry_attempts must be at least 1"):
            config = UnifiedCheckpointerConfig(retry_attempts=0)
            UnifiedCheckpointer(config)

    def test_missing_thread_id(self, checkpointer) -> None:
        """Test operations without required thread_id."""
        # Test put without thread_id
        config = {"configurable": {}}  # Missing thread_id
        checkpoint = Checkpoint(
            v=1,
            id=str(uuid.uuid4()),
            ts=datetime.now().isoformat(),
            channel_values={},
            channel_versions={},
            versions_seen={},
        )

        with pytest.raises(ConfigurationError, match="thread_id is required"):
            checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="test"),
                new_versions={},
            )

        # Test get_tuple without thread_id
        with pytest.raises(ConfigurationError, match="thread_id is required"):
            checkpointer.get_tuple(config)

        # Test put_writes without thread_id
        with pytest.raises(ConfigurationError, match="thread_id"):
            checkpointer.put_writes(config, [("channel", "value")], "task-1")

    def test_nonexistent_checkpoint(self, checkpointer) -> None:
        """Test retrieving non-existent checkpoint."""
        # Try to get checkpoint that doesn't exist
        config = {
            "configurable": {
                "thread_id": "nonexistent-thread",
                "checkpoint_id": str(uuid.uuid4()),
            },
        }

        result = checkpointer.get_tuple(config)
        assert result is None  # Should return None for non-existent checkpoint

        # Try to get latest checkpoint for thread with no checkpoints
        config_no_id = {
            "configurable": {
                "thread_id": "empty-thread",
            },
        }

        result = checkpointer.get_tuple(config_no_id)
        assert result is None  # Should return None when no checkpoints exist

    def test_empty_checkpoint_data(self, checkpointer) -> None:
        """Test handling of empty or minimal checkpoint data."""
        thread_id = "test-empty-thread"

        # Create checkpoint with minimal data
        checkpoint = Checkpoint(
            v=1,
            id=str(uuid.uuid4()),
            ts=datetime.now().isoformat(),
            channel_values={},  # Empty
            channel_versions={},  # Empty
            versions_seen={},  # Empty
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        # Should not raise error with empty data
        new_config = checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="test"),
            new_versions={},
        )

        # Should be able to retrieve it
        retrieved = checkpointer.get_tuple(new_config)
        assert retrieved is not None
        assert retrieved.checkpoint.channel_values == {}

    def test_checkpoint_without_id(self, checkpointer) -> None:
        """Test that checkpoint ID is generated if not provided."""
        thread_id = "test-no-id-thread"

        # Create checkpoint without ID
        checkpoint = {
            "v": 1,
            "ts": datetime.now().isoformat(),
            "channel_values": {"test": True},
            "channel_versions": {"test": 1},
            "versions_seen": {},
        }  # Note: no 'id' field

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        # Should generate ID automatically
        new_config = checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="test"),
            new_versions={"test": 1},
        )

        # Verify ID was generated
        assert "checkpoint_id" in new_config["configurable"]
        assert new_config["configurable"]["checkpoint_id"] is not None

        # Should be able to retrieve it
        retrieved = checkpointer.get_tuple(new_config)
        assert retrieved is not None
        assert retrieved.checkpoint.id is not None

    def test_put_writes_without_checkpoint(self, checkpointer) -> None:
        """Test put_writes for non-existent checkpoint."""
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_id": str(uuid.uuid4()),  # Non-existent
            },
        }

        # Should not raise error even if checkpoint doesn't exist
        # (pending writes can be stored independently)
        checkpointer.put_writes(
            config,
            [("channel", "value")],
            "task-1",
        )

    def test_large_checkpoint_data(self, checkpointer) -> None:
        """Test handling of large checkpoint data."""
        thread_id = "test-large-thread"

        # Create checkpoint with large data
        large_data = {"key_" + str(i): "value_" * 100 for i in range(100)}
        checkpoint = Checkpoint(
            v=1,
            id=str(uuid.uuid4()),
            ts=datetime.now().isoformat(),
            channel_values=large_data,
            channel_versions=dict.fromkeys(large_data.keys(), 1),
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        # Should handle large data without issues
        new_config = checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="test", step=1),
            new_versions=dict.fromkeys(large_data.keys(), 1),
        )

        # Should be able to retrieve it
        retrieved = checkpointer.get_tuple(new_config)
        assert retrieved is not None
        assert len(retrieved.checkpoint.channel_values) == 100

    def test_special_characters_in_ids(self, checkpointer) -> None:
        """Test handling of special characters in thread_id."""
        # Test with various special characters
        special_thread_ids = [
            "thread-with-spaces test",
            "thread/with/slashes",
            "thread:with:colons",
            "thread@with@at",
            "thread#with#hash",
        ]

        for thread_id in special_thread_ids:
            checkpoint = Checkpoint(
                v=1,
                id=str(uuid.uuid4()),
                ts=datetime.now().isoformat(),
                channel_values={"thread": thread_id},
                channel_versions={"thread": 1},
                versions_seen={},
            )

            config = {
                "configurable": {
                    "thread_id": thread_id,
                },
            }

            # Should handle special characters
            new_config = checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="test"),
                new_versions={"thread": 1},
            )

            # Should be able to retrieve it
            retrieved = checkpointer.get_tuple(new_config)
            assert retrieved is not None
            assert retrieved.checkpoint.channel_values["thread"] == thread_id
