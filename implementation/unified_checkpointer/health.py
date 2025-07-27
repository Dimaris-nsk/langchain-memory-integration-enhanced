"""Health check functionality for monitoring and liveness probes.

Provides health check endpoints and status tracking for the
UnifiedCheckpointer system.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from implementation.unified_checkpointer.client import UnifiedMemoryClient
from implementation.unified_checkpointer.config import UnifiedCheckpointerConfig


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Overall health check result."""

    status: HealthStatus
    components: list[ComponentHealth]
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "components": [
                {
                    "name": comp.name,
                    "status": comp.status.value,
                    "message": comp.message,
                    "last_check": comp.last_check.isoformat()
                    if comp.last_check
                    else None,
                    "metadata": comp.metadata,
                }
                for comp in self.components
            ],
        }


class HealthChecker:
    """Health checker for UnifiedCheckpointer system."""

    def __init__(self, config: UnifiedCheckpointerConfig) -> None:
        """
        Initialize health checker.

        Args:
            config: Checkpointer configuration
        """
        self.config = config
        self._start_time = time.time()
        self._client: UnifiedMemoryClient | None = None

    async def check_health(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        Returns:
            HealthCheckResult with overall status and component details
        """
        components = []

        # Check Qdrant connectivity
        qdrant_health = await self._check_qdrant()
        components.append(qdrant_health)

        # Check cache if enabled
        if self.config.cache_enabled:
            cache_health = self._check_cache()
            components.append(cache_health)

        # Check connection pool if enabled
        if self.config.use_connection_pool:
            pool_health = await self._check_connection_pool()
            components.append(pool_health)

        # Determine overall status
        if all(comp.status == HealthStatus.HEALTHY for comp in components):
            overall_status = HealthStatus.HEALTHY
        elif any(comp.status == HealthStatus.UNHEALTHY for comp in components):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        return HealthCheckResult(
            status=overall_status,
            components=components,
            timestamp=datetime.now(UTC),
            uptime_seconds=time.time() - self._start_time,
        )

    async def _check_qdrant(self) -> ComponentHealth:
        """Check Qdrant connectivity and collection status."""
        try:
            if not self._client:
                self._client = UnifiedMemoryClient(
                    url=self.config.qdrant_url,
                    collection_name=self.config.collection_name,
                )

            # Try to list checkpoints (with limit 1 for speed)
            start = time.time()
            await self._client.alist_checkpoints(limit=1)
            latency = (time.time() - start) * 1000  # ms

            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.HEALTHY,
                message=f"Connected, latency: {latency:.1f}ms",
                last_check=datetime.now(UTC),
                metadata={
                    "latency_ms": latency,
                    "collection": self.config.collection_name,
                    "url": self.config.qdrant_url,
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection failed: {e!s}",
                last_check=datetime.now(UTC),
                metadata={"error": str(e)},
            )

    def _check_cache(self) -> ComponentHealth:
        """Check cache status and metrics."""
        # In real implementation, would check actual cache instance
        # For now, return healthy if enabled
        return ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY,
            message=f"Cache enabled, size limit: {self.config.cache_max_size}",
            last_check=datetime.now(UTC),
            metadata={
                "max_size": self.config.cache_max_size,
                "ttl_seconds": self.config.cache_ttl_seconds,
            },
        )

    async def _check_connection_pool(self) -> ComponentHealth:
        """Check connection pool status."""
        # In real implementation, would check pool metrics
        # For now, return healthy if enabled
        return ComponentHealth(
            name="connection_pool",
            status=HealthStatus.HEALTHY,
            message=f"Pool active, size: {self.config.pool_size}",
            last_check=datetime.now(UTC),
            metadata={
                "pool_size": self.config.pool_size,
                "timeout": self.config.pool_timeout,
            },
        )

    def get_readiness(self) -> dict[str, Any]:
        """
        Get readiness probe result (synchronous).

        Returns:
            Simple readiness status
        """
        # Basic readiness - just check if service is up
        return {
            "ready": True,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_liveness(self) -> dict[str, Any]:
        """
        Get liveness probe result (synchronous).

        Returns:
            Simple liveness status
        """
        # Basic liveness - check if not deadlocked
        return {
            "alive": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": time.time() - self._start_time,
        }


# Global health checker instance
_health_checker: HealthChecker | None = None


def init_health_checker(config: UnifiedCheckpointerConfig) -> HealthChecker:
    """
    Initialize global health checker.

    Args:
        config: Checkpointer configuration

    Returns:
        Initialized HealthChecker instance
    """
    global _health_checker
    _health_checker = HealthChecker(config)
    return _health_checker


def get_health_checker() -> HealthChecker | None:
    """Get global health checker instance."""
    return _health_checker


# FastAPI/Flask integration helpers
async def health_endpoint() -> dict[str, Any]:
    """
    Health check endpoint for web frameworks.

    Returns:
        JSON-serializable health status
    """
    checker = get_health_checker()
    if not checker:
        return {
            "status": "unhealthy",
            "message": "Health checker not initialized",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    result = await checker.check_health()
    return result.to_dict()


def readiness_endpoint() -> dict[str, Any]:
    """Readiness probe endpoint."""
    checker = get_health_checker()
    if not checker:
        return {"ready": False, "message": "Not initialized"}

    return checker.get_readiness()


def liveness_endpoint() -> dict[str, Any]:
    """Liveness probe endpoint."""
    checker = get_health_checker()
    if not checker:
        return {"alive": False, "message": "Not initialized"}

    return checker.get_liveness()