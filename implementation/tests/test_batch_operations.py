"""Tests for batch operations in UnifiedCheckpointer."""

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from unified_checkpointer import UnifiedCheckpointer
from unified_checkpointer.config import UnifiedCheckpointerConfig


@pytest.fixture
def checkpointer_config():
    """Create a test configuration."""
    return UnifiedCheckpointerConfig(
        collection_name="test_batch_collection",
        unified_memory_url=None,  # Use in-memory mode
        cache_enabled=True,
        cache_max_size=10,
    )


@pytest.fixture
def batch_configs() -> list[RunnableConfig]:
    """Create a batch of test configurations."""
    return [
        {
            "configurable": {
                "thread_id": f"thread_{i}",
                "checkpoint_ns": "test_ns",
            },
        }
        for i in range(3)
    ]


@pytest.fixture
def batch_checkpoints() -> list[Checkpoint]:
    """Create a batch of test checkpoints."""
    return [
        {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": f"2023-01-0{i+1}T00:00:00.000Z",
            "channel_values": {
                "messages": [f"Message {i}"],
                "state": {"count": i},
            },
            "channel_versions": {
                "messages": i + 1,
                "state": i + 1,
            },
            "versions_seen": {},
        }
        for i in range(3)
    ]


@pytest.fixture
def batch_metadatas() -> list[CheckpointMetadata]:
    """Create a batch of test metadata."""
    return [
        {
            "source": "test",
            "step": i + 1,
            "writes": {},
            "parents": {},
        }
        for i in range(3)
    ]


@pytest.fixture
def batch_new_versions():
    """Create a batch of channel versions."""
    return [
        {
            "messages": i + 2,
            "state": i + 2,
        }
        for i in range(3)
    ]


class TestBatchOperations:
    """Test batch operations in UnifiedCheckpointer."""

    def test_put_batch_basic(self, checkpointer_config, batch_configs, batch_checkpoints, batch_metadatas, batch_new_versions) -> None:
        """Test basic batch put operation."""
        with UnifiedCheckpointer(checkpointer_config) as checkpointer:
            # Create a mock client
            mock_client = MagicMock()
            mock_client.store_checkpoint_batch = MagicMock()
            mock_client.store_checkpoint_batch.return_value = asyncio.Future()
            mock_client.store_checkpoint_batch.return_value.set_result([str(uuid.uuid4()) for _ in range(3)])

            # Mock the _get_client method to return our mock client
            with patch.object(checkpointer, "_get_client") as mock_get_client:
                mock_get_client.return_value.__enter__.return_value = mock_client

                # Call put_batch
                updated_configs = checkpointer.put_batch(
                    configs=batch_configs,
                    checkpoints=batch_checkpoints,
                    metadatas=batch_metadatas,
                    new_versions_list=batch_new_versions,
                )

                # Verify results
                assert len(updated_configs) == 3
            for i, config in enumerate(updated_configs):
                assert "checkpoint_id" in config["configurable"]
                assert config["configurable"]["thread_id"] == f"thread_{i}"

            # Verify client was called
            mock_client.store_checkpoint_batch.assert_called_once()
            call_args = mock_client.store_checkpoint_batch.call_args[0]
            assert len(call_args[0]) == 3  # 3 checkpoint data items

    def test_get_batch_basic(self, checkpointer_config, batch_configs) -> None:
        """Test basic batch get operation."""
        with UnifiedCheckpointer(checkpointer_config) as checkpointer:
            # Add checkpoint IDs to configs
            configs_with_ids = []
            checkpoint_ids = []
            for i, config in enumerate(batch_configs):
                checkpoint_id = str(uuid.uuid4())
                checkpoint_ids.append(checkpoint_id)
                config_with_id = {
                    **config,
                    "configurable": {
                        **config["configurable"],
                        "checkpoint_id": checkpoint_id,
                    },
                }
                configs_with_ids.append(config_with_id)

            # Mock the client's batch get method
            with patch.object(checkpointer._client, "get_checkpoint_batch") as mock_get:
                # Create mock checkpoint data
                mock_checkpoints = []
                for i, checkpoint_id in enumerate(checkpoint_ids):
                    mock_checkpoints.append({
                        "id": checkpoint_id,
                        "thread_id": f"thread_{i}",
                        "checkpoint": ("", '{"v": 1, "ts": "2023-01-01T00:00:00.000Z"}'),
                        "metadata": ("", '{"source": "test", "step": 1}'),
                        "pending_writes": [],
                    })

                mock_get.return_value = asyncio.Future()
                mock_get.return_value.set_result(mock_checkpoints)

                # Call get_batch
                results = checkpointer.get_batch(configs_with_ids)

                # Verify results
                assert len(results) == 3
            for i, result in enumerate(results):
                assert result is not None
                assert result.config["configurable"]["thread_id"] == f"thread_{i}"
                assert result.checkpoint["v"] == 1
                assert result.metadata["source"] == "test"

    def test_put_batch_validation(self, checkpointer_config, batch_configs, batch_checkpoints, batch_metadatas) -> None:
        """Test batch validation for mismatched input lengths."""
        with UnifiedCheckpointer(checkpointer_config) as checkpointer:
            # Try with mismatched lengths
            with pytest.raises(ValueError) as exc_info:
                checkpointer.put_batch(
                    configs=batch_configs,
                    checkpoints=batch_checkpoints[:-1],  # One less checkpoint
                    metadatas=batch_metadatas,
                    new_versions_list=[],  # Empty versions
                )

            assert "All input lists must have the same length" in str(exc_info.value)

    def test_batch_progress_callback(self, checkpointer_config, batch_configs, batch_checkpoints, batch_metadatas, batch_new_versions) -> None:
        """Test progress callback functionality."""
        with UnifiedCheckpointer(checkpointer_config) as checkpointer:
            progress_calls = []

            def progress_callback(current, total) -> None:
                progress_calls.append((current, total))

            # Mock the client's batch method
            with patch.object(checkpointer._client, "store_checkpoint_batch") as mock_store:
                # Set up the mock to call the progress callback
                def mock_store_impl(data, embeddings, progress_callback):
                    for i in range(len(data)):
                        if progress_callback:
                            progress_callback(i + 1, len(data))
                    future = asyncio.Future()
                    future.set_result([str(uuid.uuid4()) for _ in data])
                    return future

                mock_store.side_effect = mock_store_impl

                # Call put_batch with progress callback
                checkpointer.put_batch(
                    configs=batch_configs,
                    checkpoints=batch_checkpoints,
                    metadatas=batch_metadatas,
                    new_versions_list=batch_new_versions,
                    progress_callback=progress_callback,
                )

                # Verify progress was tracked
                assert len(progress_calls) == 3
                assert progress_calls == [(1, 3), (2, 3), (3, 3)]
