"""Tests for health check functionality."""
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from implementation.unified_checkpointer.config import UnifiedCheckpointerConfig
from implementation.unified_checkpointer.health import (
    ComponentHealth,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    get_health_checker,
    init_health_checker,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self) -> None:
        """Test enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_component_health_creation(self) -> None:
        """Test creating component health."""
        now = datetime.now(UTC)
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
            last_check=now,
            metadata={"key": "value"},
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.last_check == now
        assert health.metadata == {"key": "value"}

    def test_component_health_defaults(self) -> None:
        """Test default values."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.DEGRADED,
        )

        assert health.message == ""
        assert health.last_check is None
        assert health.metadata == {}


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self) -> None:
        """Test creating health check result."""
        now = datetime.now(UTC)
        components = [
            ComponentHealth("comp1", HealthStatus.HEALTHY),
            ComponentHealth("comp2", HealthStatus.DEGRADED),
        ]

        result = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            components=components,
            timestamp=now,
            version="2.0.0",
            uptime_seconds=123.45,
        )

        assert result.status == HealthStatus.DEGRADED
        assert len(result.components) == 2
        assert result.timestamp == now
        assert result.version == "2.0.0"
        assert result.uptime_seconds == 123.45

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        now = datetime.now(UTC)
        check_time = datetime.now(UTC)

        components = [
            ComponentHealth(
                name="comp1",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=check_time,
                metadata={"latency": 10},
            ),
        ]

        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            components=components,
            timestamp=now,
            uptime_seconds=100.0,
        )

        data = result.to_dict()

        assert data["status"] == "healthy"
        assert data["timestamp"] == now.isoformat()
        assert data["version"] == "1.0.0"
        assert data["uptime_seconds"] == 100.0
        assert len(data["components"]) == 1

        comp_data = data["components"][0]
        assert comp_data["name"] == "comp1"
        assert comp_data["status"] == "healthy"
        assert comp_data["message"] == "OK"
        assert comp_data["last_check"] == check_time.isoformat()
        assert comp_data["metadata"] == {"latency": 10}


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_init(self) -> None:
        """Test health checker initialization."""
        config = UnifiedCheckpointerConfig()
        checker = HealthChecker(config)

        assert checker.config == config
        assert checker._start_time > 0
        assert checker._client is None

    def test_get_readiness(self) -> None:
        """Test readiness probe."""
        config = UnifiedCheckpointerConfig()
        checker = HealthChecker(config)

        readiness = checker.get_readiness()

        assert readiness["ready"] is True
        assert "timestamp" in readiness

    def test_get_liveness(self) -> None:
        """Test liveness probe."""
        config = UnifiedCheckpointerConfig()
        checker = HealthChecker(config)

        # Let some time pass
        time.sleep(0.1)

        liveness = checker.get_liveness()

        assert liveness["alive"] is True
        assert "timestamp" in liveness
        assert liveness["uptime_seconds"] > 0


    @pytest.mark.asyncio
    async def test_check_health_all_healthy(self) -> None:
        """Test health check when all components healthy."""
        config = UnifiedCheckpointerConfig(
            cache_enabled=True,
            use_connection_pool=True,
        )
        checker = HealthChecker(config)

        # Mock the client
        mock_client = AsyncMock()
        mock_client.alist_checkpoints.return_value = []
        checker._client = mock_client

        result = await checker.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 3  # qdrant, cache, pool
        assert all(c.status == HealthStatus.HEALTHY for c in result.components)

    @pytest.mark.asyncio
    async def test_check_health_qdrant_failure(self) -> None:
        """Test health check when Qdrant fails."""
        config = UnifiedCheckpointerConfig()
        checker = HealthChecker(config)

        # Mock client creation to fail
        with patch("implementation.unified_checkpointer.health.UnifiedMemoryClient") as MockClient:
            MockClient.side_effect = Exception("Connection failed")

            result = await checker.check_health()

            assert result.status == HealthStatus.UNHEALTHY
            qdrant_comp = next(c for c in result.components if c.name == "qdrant")
            assert qdrant_comp.status == HealthStatus.UNHEALTHY
            assert "Connection failed" in qdrant_comp.message


class TestGlobalFunctions:
    """Test global health checker functions."""

    def test_init_and_get_health_checker(self) -> None:
        """Test global health checker initialization."""
        config = UnifiedCheckpointerConfig()

        # Clear any existing checker
        import implementation.unified_checkpointer.health as health_module
        health_module._health_checker = None

        # Initialize
        checker = init_health_checker(config)
        assert checker is not None
        assert isinstance(checker, HealthChecker)

        # Get should return same instance
        retrieved = get_health_checker()
        assert retrieved is checker