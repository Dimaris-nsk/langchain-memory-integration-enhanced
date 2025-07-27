"""Tests for UnifiedCheckpointer."""

import asyncio
import uuid
from datetime import datetime

import pytest
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


class TestUnifiedCheckpointer:
    """Test UnifiedCheckpointer functionality."""

    def test_put_and_get_checkpoint(self, checkpointer) -> None:
        """Test storing and retrieving a checkpoint."""
        # Prepare test data
        thread_id = "test-thread-1"
        checkpoint_id = str(uuid.uuid4())

        # Create checkpoint following LangGraph structure
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts=datetime.now().isoformat(),
            channel_values={
                "messages": ["Hello", "World"],
                "state": {"counter": 1},
            },
            channel_versions={
                "__start__": 1,
                "messages": 2,
                "state": 1,
            },
            versions_seen={
                "__input__": {},
                "__start__": {"__start__": 1},
            },
        )

        # Create metadata
        metadata = CheckpointMetadata(
            source="test",
            write_time=datetime.now().isoformat(),
            step=1,
        )

        # Create config
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test",
            },
        }

        # Save channel_versions before put (checkpoint is a dict)
        channel_versions = checkpoint["channel_versions"]

        # Put checkpoint
        new_config = checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions=channel_versions,
        )

        # Debug prints

        # Verify returned config has checkpoint_id
        assert new_config["configurable"]["checkpoint_id"] == checkpoint_id

        # Get checkpoint back
        retrieved_tuple = checkpointer.get_tuple(new_config)

        # Debug print

        # Verify retrieved data
        assert retrieved_tuple is not None
        assert isinstance(retrieved_tuple, CheckpointTuple)
        assert retrieved_tuple.checkpoint["id"] == checkpoint_id
        assert retrieved_tuple.checkpoint["channel_values"] == checkpoint["channel_values"]
        assert retrieved_tuple.metadata["source"] == "unified-memory"
        assert retrieved_tuple.config["configurable"]["thread_id"] == thread_id

    def test_get_latest_checkpoint(self, checkpointer) -> None:
        """Test retrieving latest checkpoint when no checkpoint_id specified."""
        thread_id = "test-thread-2"

        # Store multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint = Checkpoint(
                v=i + 1,
                id=str(uuid.uuid4()),
                ts=datetime.now().isoformat(),
                channel_values={"counter": i},
                channel_versions={"counter": i + 1},
                versions_seen={},
            )

            config = {
                "configurable": {
                    "thread_id": thread_id,
                },
            }

            new_config = checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="test", step=i),
                new_versions={"counter": i + 1},
            )
            checkpoint_ids.append(new_config["configurable"]["checkpoint_id"])

        # Get latest checkpoint (without specifying checkpoint_id)
        config_without_id = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        latest_tuple = checkpointer.get_tuple(config_without_id)

        # Should get the most recent checkpoint
        assert latest_tuple is not None
        assert latest_tuple.checkpoint["channel_values"]["counter"] == 2  # Last value

    def test_list_checkpoints(self, checkpointer) -> None:
        """Test listing checkpoints for a thread."""
        thread_id = "test-thread-3"

        # Store multiple checkpoints
        num_checkpoints = 5
        for i in range(num_checkpoints):
            checkpoint = Checkpoint(
                v=i + 1,
                id=str(uuid.uuid4()),
                ts=datetime.now().isoformat(),
                channel_values={"index": i},
                channel_versions={"index": i + 1},
                versions_seen={},
            )

            config = {
                "configurable": {
                    "thread_id": thread_id,
                },
            }

            checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="test", step=i),
                new_versions={"index": i + 1},
            )

        # List checkpoints
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = list(checkpointer.list(config, limit=10))

        # Verify we got all checkpoints
        assert len(checkpoints) >= num_checkpoints

        # Verify they are CheckpointTuples
        for cp in checkpoints:
            assert isinstance(cp, CheckpointTuple)
            assert cp.config["configurable"]["thread_id"] == thread_id

    def test_put_writes(self, checkpointer) -> None:
        """Test storing pending writes."""
        thread_id = "test-thread-4"
        checkpoint_id = str(uuid.uuid4())

        # First create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts=datetime.now().isoformat(),
            channel_values={},
            channel_versions={},
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            },
        }

        checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="test"),
            new_versions={},
        )

        # Now store pending writes
        task_id = str(uuid.uuid4())
        writes = [
            ("messages", ["New message"]),
            ("state", {"updated": True}),
        ]

        # This should not raise an error
        checkpointer.put_writes(config, writes, task_id)

        # Get checkpoint and verify pending writes
        tuple_with_writes = checkpointer.get_tuple(config)
        assert tuple_with_writes is not None
        assert tuple_with_writes.pending_writes is not None
        assert len(tuple_with_writes.pending_writes) == 2

        # Verify write structure
        for write in tuple_with_writes.pending_writes:
            assert len(write) == 3  # (task_id, channel, value)
            assert write[0] == task_id

    # Async tests

    @pytest.mark.asyncio
    async def test_async_put_and_get(self, checkpointer) -> None:
        """Test async storing and retrieving a checkpoint."""
        thread_id = "test-async-thread-1"
        checkpoint_id = str(uuid.uuid4())

        # Create checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts=datetime.now().isoformat(),
            channel_values={
                "messages": ["Async", "Test"],
                "state": {"async": True},
            },
            channel_versions={
                "messages": 1,
                "state": 1,
            },
            versions_seen={},
        )

        metadata = CheckpointMetadata(
            source="async-test",
            write_time=datetime.now().isoformat(),
            step=1,
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        # Save channel_versions before put (checkpoint is a dict)
        channel_versions = checkpoint["channel_versions"]

        # Async put
        new_config = await checkpointer.aput(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions=channel_versions,
        )

        # Verify returned config
        assert new_config["configurable"]["checkpoint_id"] == checkpoint_id

        # Async get
        retrieved_tuple = await checkpointer.aget_tuple(new_config)

        # Verify retrieved data
        assert retrieved_tuple is not None
        assert retrieved_tuple.checkpoint["id"] == checkpoint_id
        assert retrieved_tuple.checkpoint["channel_values"]["state"]["async"] is True

    @pytest.mark.asyncio
    async def test_async_list(self, checkpointer) -> None:
        """Test async listing of checkpoints."""
        thread_id = "test-async-thread-2"

        # Store checkpoints using async method
        for i in range(3):
            checkpoint = Checkpoint(
                v=i + 1,
                id=str(uuid.uuid4()),
                ts=datetime.now().isoformat(),
                channel_values={"async_index": i},
                channel_versions={"async_index": i + 1},
                versions_seen={},
            )

            config = {
                "configurable": {
                    "thread_id": thread_id,
                },
            }

            await checkpointer.aput(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="async-test", step=i),
                new_versions={"async_index": i + 1},
            )

        # Async list
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = []
        async for cp in checkpointer.alist(config, limit=10):
            checkpoints.append(cp)

        # Verify results
        assert len(checkpoints) >= 3
        for cp in checkpoints:
            assert isinstance(cp, CheckpointTuple)
            assert cp.config["configurable"]["thread_id"] == thread_id

    @pytest.mark.asyncio
    async def test_async_put_writes(self, checkpointer) -> None:
        """Test async storing of pending writes."""
        thread_id = "test-async-thread-3"
        checkpoint_id = str(uuid.uuid4())

        # Create checkpoint first
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts=datetime.now().isoformat(),
            channel_values={},
            channel_versions={},
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            },
        }

        await checkpointer.aput(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="async-test"),
            new_versions={},
        )

        # Store pending writes asynchronously
        task_id = str(uuid.uuid4())
        writes = [
            ("async_channel", {"async": True}),
            ("messages", ["Async write"]),
        ]

        await checkpointer.aput_writes(config, writes, task_id)

        # Verify writes were stored
        tuple_with_writes = await checkpointer.aget_tuple(config)
        assert tuple_with_writes is not None
        assert tuple_with_writes.pending_writes is not None
        assert len(tuple_with_writes.pending_writes) == 2


# Pytest async test runner
def test_async_methods(checkpointer) -> None:
    """Run all async tests."""
    test_instance = TestUnifiedCheckpointer()

    # Run async tests
    asyncio.run(test_instance.test_async_put_and_get(checkpointer))
    asyncio.run(test_instance.test_async_list(checkpointer))
    asyncio.run(test_instance.test_async_put_writes(checkpointer))