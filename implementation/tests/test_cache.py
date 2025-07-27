"""Tests for LRU cache implementation."""

import time

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from unified_checkpointer.cache import CheckpointCache


@pytest.fixture
def cache():
    """Create a cache instance for testing."""
    return CheckpointCache(max_size=10, ttl_seconds=300)


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    return CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_id": "test-checkpoint-1",
            },
        },
        checkpoint=Checkpoint(
            v=1,
            id="test-checkpoint-1",
            ts="2025-01-01T00:00:00+00:00",
            channel_values={},
            channel_versions={},
            versions_seen={},
        ),
        metadata=CheckpointMetadata(),
        parent_config=None,
        pending_writes=None,
    )


class TestCheckpointCache:
    """Test suite for CheckpointCache."""

    def test_cache_basic_operations(self, cache, sample_checkpoint) -> None:
        """Test basic get/put operations."""
        thread_id = "test-thread"
        checkpoint_id = "test-checkpoint-1"

        # Initially cache should be empty
        assert cache.get(thread_id, checkpoint_id) is None
        assert cache.get_stats()["misses"] == 1

        # Put checkpoint in cache
        cache.put(thread_id, sample_checkpoint, checkpoint_id)

        # Should be able to retrieve it
        cached = cache.get(thread_id, checkpoint_id)
        assert cached is not None
        assert cached.checkpoint.id == checkpoint_id
        assert cache.get_stats()["hits"] == 1

        # Should also be cached as "latest"
        latest = cache.get(thread_id)
        assert latest is not None
        assert latest.checkpoint.id == checkpoint_id

    def test_cache_ttl_expiration(self, sample_checkpoint) -> None:
        """Test TTL expiration functionality."""
        # Create cache with 1 second TTL
        cache = CheckpointCache(max_size=10, ttl_seconds=1)
        thread_id = "test-thread"
        checkpoint_id = "test-checkpoint-1"

        # Put checkpoint
        cache.put(thread_id, sample_checkpoint, checkpoint_id)

        # Should be retrievable immediately
        assert cache.get(thread_id, checkpoint_id) is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired now
        assert cache.get(thread_id, checkpoint_id) is None
        assert cache.get_stats()["expirations"] == 1

    def test_cache_invalidation(self, cache, sample_checkpoint) -> None:
        """Test cache invalidation."""
        thread_id = "test-thread"
        checkpoint_id = "test-checkpoint-1"

        # Put checkpoint
        cache.put(thread_id, sample_checkpoint, checkpoint_id)
        assert cache.get(thread_id, checkpoint_id) is not None

        # Invalidate specific checkpoint
        cache.invalidate(thread_id, checkpoint_id)
        assert cache.get(thread_id, checkpoint_id) is None

        # Put another checkpoint
        cache.put(thread_id, sample_checkpoint, checkpoint_id)
        cache.put(thread_id, sample_checkpoint, "test-checkpoint-2")

        # Invalidate all for thread
        cache.invalidate(thread_id)
        assert cache.get(thread_id, checkpoint_id) is None
        assert cache.get(thread_id, "test-checkpoint-2") is None

    def test_cache_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = CheckpointCache(max_size=3, ttl_seconds=300)

        # Fill cache
        for i in range(3):
            checkpoint = CheckpointTuple(
                config={"configurable": {"thread_id": f"thread-{i}", "checkpoint_id": f"cp-{i}"}},
                checkpoint=Checkpoint(v=1, id=f"cp-{i}", ts="2025-01-01T00:00:00+00:00",
                                    channel_values={}, channel_versions={}, versions_seen={}),
                metadata=CheckpointMetadata(),
                parent_config=None,
                pending_writes=None,
            )
            cache.put(f"thread-{i}", checkpoint, f"cp-{i}")

        # Add one more - should evict the oldest
        checkpoint = CheckpointTuple(
            config={"configurable": {"thread_id": "thread-3", "checkpoint_id": "cp-3"}},
            checkpoint=Checkpoint(v=1, id="cp-3", ts="2025-01-01T00:00:00+00:00",
                                channel_values={}, channel_versions={}, versions_seen={}),
            metadata=CheckpointMetadata(),
            parent_config=None,
            pending_writes=None,
        )
        cache.put("thread-3", checkpoint, "cp-3")

        # Oldest should be evicted
        assert cache.get("thread-0", "cp-0") is None
        assert cache.get_stats()["evictions"] >= 1

        # Others should still be there
        assert cache.get("thread-1", "cp-1") is not None
        assert cache.get("thread-2", "cp-2") is not None
        assert cache.get("thread-3", "cp-3") is not None

    def test_cache_stats(self, cache, sample_checkpoint) -> None:
        """Test cache statistics tracking."""
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Miss
        cache.get("thread-1", "cp-1")
        stats = cache.get_stats()
        assert stats["misses"] == 1

        # Hit
        cache.put("thread-1", sample_checkpoint, "cp-1")
        cache.get("thread-1", "cp-1")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 0.5  # 1 hit, 2 total requests
