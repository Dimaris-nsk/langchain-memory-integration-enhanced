"""Pytest configuration and fixtures."""

import pytest
from unified_checkpointer import UnifiedCheckpointer, UnifiedCheckpointerConfig
from unified_checkpointer.client import UnifiedMemoryClient


@pytest.fixture
def config():
    """Create a test configuration."""
    return UnifiedCheckpointerConfig(
        collection_name="test_checkpoints",
        unified_memory_url=None,  # Use in-memory mode
        enable_search=True,
        enable_analytics=True,
        cache_size=10,
        connection_pool_size=1,  # Minimal pool size for tests
    )


@pytest.fixture
def client():
    """Create an in-memory test client."""
    return UnifiedMemoryClient(
        collection_name="test_checkpoints",
        url=None,  # Use in-memory mode
    )


@pytest.fixture
def checkpointer(config, client):
    """Create a test checkpointer and properly clean it up."""
    # Pass the same client to checkpointer for shared in-memory storage
    cp = UnifiedCheckpointer(config, client=client)
    yield cp
    # Cleanup: close the checkpointer and its connection pool
    cp.close()


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    return {
        "thread_id": "test-thread-1",
        "checkpoint_ns": "test",
        "checkpoint": {
            "v": 1,
            "ts": "2025-07-22T20:00:00Z",
            "channel_values": {"messages": ["Hello", "World"]},
            "channel_versions": {"messages": 2},
        },
        "metadata": {"source": "test", "user": "test-user"},
        "tags": ["test", "sample"],
    }
