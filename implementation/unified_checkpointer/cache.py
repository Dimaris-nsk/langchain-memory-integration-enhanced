"""LRU Cache implementation for UnifiedCheckpointer.

This module provides caching functionality to improve performance
by avoiding repeated fetches of frequently accessed checkpoints.
"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import CheckpointTuple


@dataclass
class CacheEntry:
    """Represents a cached checkpoint entry."""

    checkpoint: CheckpointTuple
    timestamp: datetime
    access_count: int = 0

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry has expired."""
        if ttl_seconds <= 0:  # No expiry
            return False
        expiry_time = self.timestamp + timedelta(seconds=ttl_seconds)
        return datetime.now(UTC) > expiry_time


class CheckpointCache:
    """LRU cache for checkpoint storage with TTL support.

    This cache improves performance by storing frequently accessed
    checkpoints in memory, reducing the need for database queries.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds (0 for no expiry)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}

    def _make_key(self, thread_id: str, checkpoint_id: str | None = None) -> str:
        """Create a cache key from thread and checkpoint IDs."""
        if checkpoint_id:
            return f"{thread_id}:{checkpoint_id}"
        return f"{thread_id}:latest"

    def get(
        self, thread_id: str, checkpoint_id: str | None = None,
    ) -> CheckpointTuple | None:
        """Get a checkpoint from cache.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID (None for latest)

        Returns:
            Cached checkpoint or None if not found/expired
        """
        key = self._make_key(thread_id, checkpoint_id)

        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired(self.ttl_seconds):
            del self._cache[key]
            self._stats["expirations"] += 1
            self._stats["misses"] += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.access_count += 1
        self._stats["hits"] += 1

        return entry.checkpoint

    def put(
        self,
        thread_id: str,
        checkpoint: CheckpointTuple,
        checkpoint_id: str | None = None,
    ) -> None:
        """Store a checkpoint in cache.

        Args:
            thread_id: Thread identifier
            checkpoint: Checkpoint to cache
            checkpoint_id: Specific checkpoint ID (None for latest)
        """
        key = self._make_key(thread_id, checkpoint_id or checkpoint.checkpoint["id"])

        # Store with specific ID
        specific_entry = CacheEntry(checkpoint=checkpoint, timestamp=datetime.now(UTC))
        self._cache[key] = specific_entry
        self._cache.move_to_end(key)

        # Also update "latest" entry if this is the newest
        latest_key = self._make_key(thread_id)
        if latest_key in self._cache:
            latest_entry = self._cache[latest_key]
            if checkpoint.checkpoint["ts"] > latest_entry.checkpoint.checkpoint["ts"]:
                self._cache[latest_key] = specific_entry
        else:
            self._cache[latest_key] = specific_entry

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            evicted_key = next(iter(self._cache))
            del self._cache[evicted_key]
            self._stats["evictions"] += 1

    def invalidate(self, thread_id: str, checkpoint_id: str | None = None) -> None:
        """Remove entries from cache.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID or None to clear all for thread
        """
        if checkpoint_id:
            # Invalidate specific checkpoint
            key = self._make_key(thread_id, checkpoint_id)
            self._cache.pop(key, None)
        else:
            # Invalidate all checkpoints for thread
            keys_to_remove = [
                k for k in self._cache if k.startswith(f"{thread_id}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def warm_up(self, checkpoints: list[tuple[str, CheckpointTuple]]) -> None:
        """Pre-populate cache with checkpoints.

        Args:
            checkpoints: List of (thread_id, checkpoint) tuples
        """
        for thread_id, checkpoint in checkpoints:
            self.put(thread_id, checkpoint)