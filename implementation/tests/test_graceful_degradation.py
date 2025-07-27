"""
Comprehensive tests for graceful degradation system.

Tests all fallback strategies, degradation coordinator, and checkpointer wrapper
to ensure production-ready resilience and fault tolerance.
"""

import os

# Import components under test
import sys
import time
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from unified_checkpointer.graceful_degradation import (
    CheckpointerWithDegradation,
    DegradationConfig,
    DirectStorageFallback,
    FallbackStrategy,
    GracefulDegradation,
    InMemoryQdrantFallback,
    NoOpEmbeddingsFallback,
    ServiceLevel,
    SingleConnectionFallback,
)


class TestFallbackStrategies:
    """Test all fallback strategy implementations."""

    def test_inmemory_qdrant_fallback(self) -> None:
        """Test in-memory Qdrant fallback functionality."""
        fallback = InMemoryQdrantFallback()

        # Test availability
        assert fallback.is_available()

        # Test store operation
        result = fallback.execute(
            "store_checkpoint",
            checkpoint_id="test_1",
            data={"thread_id": "thread_1", "content": "test data"},
        )
        assert result

        # Test retrieve operation
        retrieved = fallback.execute(
            "get_checkpoint",
            checkpoint_id="test_1",
        )
        assert retrieved == {"thread_id": "thread_1", "content": "test data"}

        # Test list operation
        listed = fallback.execute(
            "list_checkpoints",
            thread_id="thread_1",
        )
        assert len(listed) == 1
        assert listed[0]["thread_id"] == "thread_1"

        # Test activation/deactivation
        fallback.on_activate()  # Should not raise
        fallback.on_deactivate()  # Should clear storage

        # Verify storage cleared
        retrieved_after_clear = fallback.execute(
            "get_checkpoint",
            checkpoint_id="test_1",
        )
        assert retrieved_after_clear is None

    def test_noop_embeddings_fallback(self) -> None:
        """Test no-op embeddings fallback functionality."""
        fallback = NoOpEmbeddingsFallback()

        # Test availability
        assert fallback.is_available()

        # Test generate_embedding
        result = fallback.execute("generate_embedding", text="test")
        assert result is None

        # Test embed_documents
        result = fallback.execute("embed_documents", texts=["doc1", "doc2"])
        assert result == [None, None]

        # Test embed_query
        result = fallback.execute("embed_query", query="test query")
        assert result is None

        # Test semantic_search
        result = fallback.execute("semantic_search", query="test")
        assert result == []

        # Test unknown operation
        result = fallback.execute("unknown_operation")
        assert result is None

    def test_direct_storage_fallback(self) -> None:
        """Test direct storage fallback functionality."""
        # Mock client
        mock_client = Mock()
        mock_client.get_checkpoint.return_value = {"data": "test"}
        mock_client.store_checkpoint.return_value = True
        mock_client.list_checkpoints.return_value = [{"id": "1"}]

        fallback = DirectStorageFallback(mock_client)

        # Test availability
        assert fallback.is_available()

        # Test get_checkpoint
        result = fallback.execute("get_checkpoint", checkpoint_id="test_1")
        assert result == {"data": "test"}
        mock_client.get_checkpoint.assert_called_with("test_1")

        # Test store_checkpoint
        result = fallback.execute("store_checkpoint", data="test")
        assert result
        mock_client.store_checkpoint.assert_called_with(data="test")

        # Test list_checkpoints
        result = fallback.execute("list_checkpoints")
        assert result == [{"id": "1"}]
        mock_client.list_checkpoints.assert_called_once()

    def test_single_connection_fallback(self) -> None:
        """Test single connection fallback functionality."""
        # Mock client creation function
        mock_client = Mock()
        mock_client.some_operation.return_value = "success"

        def create_client():
            return mock_client

        fallback = SingleConnectionFallback(create_client)

        # Test availability
        assert fallback.is_available()

        # Test operation execution
        result = fallback.execute("some_operation", arg1="test")
        assert result == "success"
        mock_client.some_operation.assert_called_with(arg1="test")

        # Test deactivation with client cleanup
        mock_client.close = Mock()
        fallback.on_deactivate()
        mock_client.close.assert_called_once()


class TestGracefulDegradation:
    """Test graceful degradation coordinator."""

    @pytest.fixture
    def degradation_config(self):
        """Create test configuration."""
        return DegradationConfig(
            health_check_interval=1,  # Fast for testing
            max_error_threshold=3,
            recovery_timeout=5,
            automatic_recovery=False,  # Disable for controlled testing
        )

    @pytest.fixture
    def degradation(self, degradation_config):
        """Create degradation instance."""
        return GracefulDegradation(degradation_config)

    def test_component_registration(self, degradation) -> None:
        """Test component registration and status tracking."""
        # Register component
        degradation.register_component("test_component")

        # Verify registration
        assert "test_component" in degradation.components
        component = degradation.components["test_component"]
        assert component.name == "test_component"
        assert component.is_healthy
        assert component.error_count == 0

    def test_fallback_registration(self, degradation) -> None:
        """Test fallback strategy registration."""
        mock_strategy = Mock(spec=FallbackStrategy)
        mock_strategy.is_available.return_value = True

        # Register fallback
        degradation.register_fallback("test_fallback", mock_strategy)

        # Verify registration
        assert "test_fallback" in degradation.fallbacks
        assert degradation.fallbacks["test_fallback"] == mock_strategy

    def test_service_level_updates(self, degradation) -> None:
        """Test service level calculation."""
        # No components - should be FULL
        degradation.update_service_level()
        assert degradation.service_level == ServiceLevel.FULL

        # All healthy components
        degradation.register_component("comp1")
        degradation.register_component("comp2")
        degradation.update_service_level()
        assert degradation.service_level == ServiceLevel.FULL

        # Some unhealthy components
        degradation.components["comp1"].is_healthy = False
        degradation.update_service_level()
        assert degradation.service_level == ServiceLevel.DEGRADED

        # All unhealthy components
        degradation.components["comp2"].is_healthy = False
        degradation.update_service_level()
        assert degradation.service_level == ServiceLevel.OFFLINE

    def test_fallback_activation_deactivation(self, degradation) -> None:
        """Test fallback activation and deactivation."""
        mock_strategy = Mock(spec=FallbackStrategy)
        mock_strategy.is_available.return_value = True

        degradation.register_fallback("test_fallback", mock_strategy)

        # Test activation
        degradation.activate_fallback("test_fallback")
        assert "test_fallback" in degradation.active_fallbacks
        mock_strategy.on_activate.assert_called_once()

        # Test deactivation
        degradation.deactivate_fallback("test_fallback")
        assert "test_fallback" not in degradation.active_fallbacks
        mock_strategy.on_deactivate.assert_called_once()

    def test_circuit_breaker(self, degradation) -> None:
        """Test circuit breaker functionality."""
        # Initially closed
        assert not degradation.is_circuit_open()

        # Record multiple failures
        for _ in range(degradation.config.circuit_breaker_threshold):
            degradation.record_operation(success=False)

        # Should be open now
        assert degradation.is_circuit_open()
        assert degradation.metrics["circuit_breaker_trips"] == 1

        # Record success to start recovery
        degradation.record_operation(success=True)
        # Circuit should move to HALF_OPEN after timeout (simulated)
        degradation._circuit_last_failure_time = time.time() - degradation.config.recovery_timeout - 1
        degradation.record_operation(success=True)
        assert not degradation.is_circuit_open()

    def test_metrics_tracking(self, degradation) -> None:
        """Test metrics collection."""
        # Initial state
        metrics = degradation.get_metrics()
        assert metrics["total_operations"] == 0
        assert metrics["failed_operations"] == 0

        # Record operations
        degradation.record_operation(success=True)
        degradation.record_operation(success=False)

        # Check updated metrics
        metrics = degradation.get_metrics()
        assert metrics["total_operations"] == 2
        assert metrics["failed_operations"] == 1


class TestCheckpointerWithDegradation:
    """Test checkpointer wrapper with degradation support."""

    @pytest.fixture
    def mock_checkpointer(self):
        """Create mock checkpointer."""
        checkpointer = Mock()
        checkpointer.put.return_value = "put_result"
        checkpointer.get_tuple.return_value = "get_result"
        checkpointer.list.return_value = ["list_result"]
        checkpointer.put_writes.return_value = "writes_result"

        # Async methods
        async def mock_aput(*args, **kwargs) -> str:
            return "aput_result"
        async def mock_aget_tuple(*args, **kwargs) -> str:
            return "aget_result"
        async def mock_alist(*args, **kwargs):
            return ["alist_result"]
        async def mock_aput_writes(*args, **kwargs) -> str:
            return "awrites_result"

        checkpointer.aput = mock_aput
        checkpointer.aget_tuple = mock_aget_tuple
        checkpointer.alist = mock_alist
        checkpointer.aput_writes = mock_aput_writes

        return checkpointer

    @pytest.fixture
    def wrapper(self, mock_checkpointer):
        """Create checkpointer wrapper."""
        config = DegradationConfig(automatic_recovery=False)
        degradation = GracefulDegradation(config)
        return CheckpointerWithDegradation(mock_checkpointer, degradation)

    def test_sync_operations_success(self, wrapper, mock_checkpointer) -> None:
        """Test successful sync operations."""
        # Test put
        result = wrapper.put("config", "checkpoint", "metadata")
        assert result == "put_result"
        mock_checkpointer.put.assert_called_with("config", "checkpoint", "metadata")

        # Test get_tuple
        result = wrapper.get_tuple("config")
        assert result == "get_result"
        mock_checkpointer.get_tuple.assert_called_with("config")

        # Test list
        result = wrapper.list("config")
        assert result == ["list_result"]
        mock_checkpointer.list.assert_called_with("config")

        # Test put_writes
        result = wrapper.put_writes("config", "writes")
        assert result == "writes_result"
        mock_checkpointer.put_writes.assert_called_with("config", "writes")

    @pytest.mark.asyncio
    async def test_async_operations_success(self, wrapper, mock_checkpointer) -> None:
        """Test successful async operations."""
        # Test aput
        result = await wrapper.aput("config", "checkpoint", "metadata")
        assert result == "aput_result"

        # Test aget_tuple
        result = await wrapper.aget_tuple("config")
        assert result == "aget_result"

        # Test alist
        result = await wrapper.alist("config")
        assert result == ["alist_result"]

        # Test aput_writes
        result = await wrapper.aput_writes("config", "writes")
        assert result == "awrites_result"

    def test_error_handling_and_metrics(self, wrapper, mock_checkpointer) -> None:
        """Test error handling and metrics recording."""
        # Make operation fail
        mock_checkpointer.put.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            wrapper.put("config", "checkpoint", "metadata")

        # Check that error was recorded
        metrics = wrapper.degradation.get_metrics()
        assert metrics["failed_operations"] == 1
        assert metrics["total_operations"] == 1

    def test_service_status(self, wrapper) -> None:
        """Test service status reporting."""
        status = wrapper.get_service_status()

        # Check required fields
        assert "service_level" in status
        assert "components" in status
        assert "active_fallbacks" in status
        assert "metrics" in status
        assert "circuit_breaker" in status

        # Check component status structure
        for component_status in status["components"].values():
            assert "healthy" in component_status
            assert "error_count" in component_status
            assert "last_error" in component_status
            assert "last_check" in component_status
            assert "recovery_attempts" in component_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
