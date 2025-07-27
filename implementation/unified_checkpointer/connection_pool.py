"""Connection pooling for Qdrant client connections."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

from .client import UnifiedMemoryClient
from .exceptions import ConnectionError

logger = logging.getLogger(__name__)


class ConnectionStats:
    """Statistics for connection pool monitoring."""

    def __init__(self) -> None:
        self.total_connections_created = 0
        self.total_connections_closed = 0
        self.total_acquisitions = 0
        self.total_releases = 0
        self.current_active = 0
        self.current_idle = 0
        self.health_check_failures = 0
        self.connection_errors = 0
        self.wait_time_total = 0.0
        self.wait_time_count = 0

    @property
    def average_wait_time(self) -> float:
        """Get average wait time for connection acquisition."""
        if self.wait_time_count == 0:
            return 0.0
        return self.wait_time_total / self.wait_time_count

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_connections_created": self.total_connections_created,
            "total_connections_closed": self.total_connections_closed,
            "total_acquisitions": self.total_acquisitions,
            "total_releases": self.total_releases,
            "current_active": self.current_active,
            "current_idle": self.current_idle,
            "health_check_failures": self.health_check_failures,
            "connection_errors": self.connection_errors,
            "average_wait_time": self.average_wait_time,
        }


class ConnectionWrapper:
    """Wrapper for Qdrant client connection with metadata."""

    def __init__(self, client: Any, is_async: bool = False) -> None:
        self.client = client
        self.is_async = is_async
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.use_count = 0
        self.is_healthy = True
        self._lock = asyncio.Lock()

    def mark_used(self) -> None:
        """Mark connection as used."""
        self.last_used_at = time.time()
        self.use_count += 1

    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used_at


class ConnectionPool:
    """Async connection pool for Qdrant clients.

    This pool manages both sync and async Qdrant client connections,
    providing features like:
    - Connection lifecycle management
    - Health checking
    - Connection recycling
    - Statistics and monitoring
    """

    def __init__(
        self,
        collection_name: str,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int = 768,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_ttl: float = 3600.0,  # 1 hour
        idle_timeout: float = 600.0,  # 10 minutes
        health_check_interval: float = 60.0,
        is_async: bool = True,
        client: UnifiedMemoryClient | None = None,  # Add client parameter
    ) -> None:
        """Initialize connection pool.

        Args:
            collection_name: Qdrant collection name
            url: Qdrant server URL
            api_key: API key for authentication
            vector_size: Vector embedding dimension
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            connection_ttl: Maximum age of a connection in seconds
            idle_timeout: Maximum idle time before closing connection
            health_check_interval: Interval between health checks
            is_async: Whether to create async clients
            client: Optional pre-configured UnifiedMemoryClient to use instead of creating new ones
        """
        self.collection_name = collection_name
        self.url = url if url != ":memory:" else ":memory:"
        self.api_key = api_key
        self.vector_size = vector_size
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_ttl = connection_ttl
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        self.is_async = is_async
        self._provided_client = client  # Store provided client

        # Pool state
        self._pool: asyncio.Queue[ConnectionWrapper] = asyncio.Queue(
            maxsize=max_connections,
        )
        self._active_connections: set[ConnectionWrapper] = set()
        self._all_connections: set[ConnectionWrapper] = set()
        self._stats = ConnectionStats()
        self._closed = False
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._closed:
            msg = "Connection pool is closed"
            raise ConnectionError(msg)

        # Create minimum connections
        for _ in range(self.min_connections):
            try:
                conn = await self._create_connection()
                await self._pool.put(conn)
                self._stats.current_idle += 1
            except Exception as e:
                logger.exception(f"Failed to create initial connection: {e}")
                self._stats.connection_errors += 1

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _create_connection(self) -> ConnectionWrapper:
        """Create a new Qdrant connection."""
        try:
            # Use provided client if available, otherwise create new one
            if self._provided_client:
                client = self._provided_client
            else:
                # Create UnifiedMemoryClient which handles both sync and async internally
                client = UnifiedMemoryClient(
                    collection_name=self.collection_name,
                    url=self.url,
                    api_key=self.api_key,
                    vector_size=self.vector_size,
                )

            # No need to ensure collection - UnifiedMemoryClient does it in __init__

            conn = ConnectionWrapper(client, is_async=self.is_async)
            self._all_connections.add(conn)
            self._stats.total_connections_created += 1

            logger.debug(
                f"Created new connection (total: {len(self._all_connections)})",
            )
            return conn

        except Exception as e:
            self._stats.connection_errors += 1
            msg = f"Failed to create connection: {e}"
            raise ConnectionError(msg)

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Any, None]:
        """Acquire a connection from the pool.

        Usage:
            async with pool.acquire() as client:
                # Use client
                pass
        """
        if self._closed:
            msg = "Connection pool is closed"
            raise ConnectionError(msg)

        start_time = time.time()
        conn: ConnectionWrapper | None = None

        try:
            # Try to get from pool
            try:
                conn = await asyncio.wait_for(self._pool.get(), timeout=0.1)
                self._stats.current_idle -= 1
            except TimeoutError:
                # Pool is empty, create new if under limit
                async with self._lock:
                    if len(self._all_connections) < self.max_connections:
                        conn = await self._create_connection()
                    else:
                        # Wait for available connection
                        conn = await self._pool.get()
                        self._stats.current_idle -= 1

            # Update stats
            wait_time = time.time() - start_time
            self._stats.wait_time_total += wait_time
            self._stats.wait_time_count += 1
            self._stats.total_acquisitions += 1
            self._stats.current_active += 1

            # Mark as active
            conn.mark_used()
            self._active_connections.add(conn)

            yield conn.client

        finally:
            if conn:
                # Return to pool
                self._active_connections.discard(conn)
                self._stats.current_active -= 1

                # Check if connection should be recycled
                if (
                    conn.age() > self.connection_ttl
                    or not conn.is_healthy
                    or self._closed
                ):
                    await self._close_connection(conn)
                else:
                    # Return to pool
                    try:
                        self._pool.put_nowait(conn)
                        self._stats.current_idle += 1
                        self._stats.total_releases += 1
                    except asyncio.QueueFull:
                        # Pool is full, close connection
                        await self._close_connection(conn)

    async def _close_connection(self, conn: ConnectionWrapper) -> None:
        """Close a connection."""
        try:
            self._all_connections.discard(conn)
            self._stats.total_connections_closed += 1

            # Close client if it has close method
            if hasattr(conn.client, "close"):
                if conn.is_async:
                    await conn.client.close()
                else:
                    conn.client.close()

            logger.debug(f"Closed connection (remaining: {len(self._all_connections)})")

        except Exception as e:
            logger.exception(f"Error closing connection: {e}")

    async def _health_check_loop(self) -> None:
        """Background task for health checking connections."""
        while not self._closed:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_pool_health()
            except Exception as e:
                logger.exception(f"Health check error: {e}")

    async def _check_pool_health(self) -> None:
        """Check health of idle connections."""
        idle_connections = []

        # Get all idle connections
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                idle_connections.append(conn)
            except asyncio.QueueEmpty:
                break

        # Check each connection
        for conn in idle_connections:
            try:
                # Check if connection is too old or idle
                if (
                    conn.age() > self.connection_ttl
                    or conn.idle_time() > self.idle_timeout
                ):
                    await self._close_connection(conn)
                    self._stats.current_idle -= 1
                    continue

                # Perform health check
                if await self._check_connection_health(conn):
                    conn.is_healthy = True
                    self._pool.put_nowait(conn)
                else:
                    conn.is_healthy = False
                    await self._close_connection(conn)
                    self._stats.current_idle -= 1
                    self._stats.health_check_failures += 1

            except Exception as e:
                logger.exception(f"Error checking connection health: {e}")
                await self._close_connection(conn)
                self._stats.current_idle -= 1

        # Ensure minimum connections
        async with self._lock:
            current_total = len(self._all_connections)
            if current_total < self.min_connections:
                for _ in range(self.min_connections - current_total):
                    try:
                        conn = await self._create_connection()
                        await self._pool.put(conn)
                        self._stats.current_idle += 1
                    except Exception as e:
                        logger.exception(f"Failed to create connection: {e}")

    async def _check_connection_health(self, conn: ConnectionWrapper) -> bool:
        """Check if a connection is healthy."""
        try:
            # Simple health check - try to get collections
            if conn.is_async and hasattr(conn.client, "get_collections"):
                await conn.client.get_collections()
            else:
                conn.client.get_collections()
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return self._stats.to_dict()

    async def close(self) -> None:
        """Close all connections and shut down the pool."""
        self._closed = True

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close all connections
        all_conns = list(self._all_connections)
        for conn in all_conns:
            await self._close_connection(conn)

        logger.info(f"Connection pool closed. Stats: {self.get_stats()}")

    # Synchronous wrapper methods for non-async contexts

    def acquire_sync(self):
        """Synchronous wrapper for acquire().

        This method allows using the connection pool from synchronous code.
        It handles the async context properly without creating new event loops.

        Usage:
            with pool.acquire_sync() as client:
                # Use client
                pass
        """
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an async context but called from sync code
            # This is tricky - we need to use a different approach
            msg = "Cannot use acquire_sync from async context"
            raise RuntimeError(msg)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            pass

        # Use a helper to manage the async context
        class SyncConnectionContext:
            def __init__(self, pool) -> None:
                self.pool = pool
                self.client = None
                self.context = None

            def __enter__(self):
                # Run the async acquisition in a new event loop
                async def _acquire():
                    self.context = self.pool.acquire()
                    return await self.context.__aenter__()

                self.client = asyncio.run(_acquire())
                return self.client

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Run the async cleanup
                async def _release() -> None:
                    await self.context.__aexit__(exc_type, exc_val, exc_tb)

                asyncio.run(_release())

        return SyncConnectionContext(self)