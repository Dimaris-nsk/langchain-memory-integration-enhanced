# Graceful Degradation System for UnifiedCheckpointer
"""
Production-ready graceful degradation system that provides fallback strategies
when primary services (Qdrant, embeddings, cache) become unavailable.

This module implements the Strategy pattern for different fallback behaviors
and provides a unified interface for handling service degradation scenarios.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ServiceLevel(Enum):
    """Service level definitions for graceful degradation."""

    FULL = "full"  # All services operational
    DEGRADED = "degraded"  # Some services down, fallbacks active
    MINIMAL = "minimal"  # Critical services only
    OFFLINE = "offline"  # All services down


@dataclass
class ComponentStatus:
    """Status tracking for individual components."""

    name: str
    is_healthy: bool = True
    last_check: float = field(default_factory=time.time)
    error_count: int = 0
    last_error: str | None = None
    recovery_attempts: int = 0
    last_recovery_attempt: float = 0


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation system."""

    health_check_interval: int = 30  # seconds
    max_error_threshold: int = 5
    recovery_timeout: int = 300  # seconds
    max_recovery_attempts: int = 3
    circuit_breaker_threshold: int = 10
    enabled_fallbacks: list[str] = field(
        default_factory=lambda: [
            "InMemoryQdrantFallback",
            "NoOpEmbeddingsFallback",
            "DirectStorageFallback",
            "SingleConnectionFallback",
        ],
    )
    automatic_recovery: bool = True
    metrics_enabled: bool = True


class FallbackStrategy(ABC):
    """Base class for fallback strategies."""

    @abstractmethod
    def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute the fallback operation."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this fallback is available."""

    def on_activate(self) -> None:
        """Called when fallback is activated."""

    def on_deactivate(self) -> None:
        """Called when fallback is deactivated."""


class InMemoryQdrantFallback(FallbackStrategy):
    """In-memory fallback for Qdrant operations."""

    def __init__(self) -> None:
        # Use thread-local storage for thread safety
        self._local = threading.local()
        self.logger = logging.getLogger(f"{__name__}.InMemoryQdrantFallback")

    def _get_storage(self) -> dict[str, Any]:
        """Get thread-local storage."""
        if not hasattr(self._local, "storage"):
            self._local.storage = {}
        return self._local.storage

    def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute in-memory operations."""
        storage = self._get_storage()

        if operation == "store_checkpoint":
            checkpoint_id = kwargs.get("checkpoint_id")
            data = kwargs.get("data")
            storage[checkpoint_id] = data
            self.logger.info(f"Stored checkpoint {checkpoint_id} in memory")
            return True

        if operation == "get_checkpoint":
            checkpoint_id = kwargs.get("checkpoint_id")
            return storage.get(checkpoint_id)

        if operation == "list_checkpoints":
            thread_id = kwargs.get("thread_id")
            # Simple filtering by thread_id if stored in data
            results = []
            for data in storage.values():
                if data.get("thread_id") == thread_id:
                    results.append(data)
            return results

        self.logger.warning(f"Unknown operation: {operation}")
        return None

    def is_available(self) -> bool:
        """In-memory fallback is always available."""
        return True

    def on_activate(self) -> None:
        """Log activation."""
        self.logger.warning("Activated in-memory fallback for Qdrant")

    def on_deactivate(self) -> None:
        """Clear memory on deactivation."""
        if hasattr(self._local, "storage"):
            self._local.storage.clear()
        self.logger.info("Deactivated in-memory fallback")


class NoOpEmbeddingsFallback(FallbackStrategy):
    """No-operation fallback for embeddings when service is down."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.NoOpEmbeddingsFallback")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute no-op embedding operations."""
        if operation == "generate_embedding":
            # Return None or empty vector to disable semantic search
            return None

        if operation == "embed_documents":
            texts = kwargs.get("texts", args[0] if args else [])
            # Return list of None for each text
            return [None] * len(texts)

        if operation == "embed_query":
            # Return None to disable semantic search
            return None

        if operation == "semantic_search":
            # Return empty results when embeddings disabled
            return []

        self.logger.warning(f"Unknown embedding operation: {operation}")
        return None

    def is_available(self) -> bool:
        """NoOp fallback is always available."""
        return True

    def on_activate(self) -> None:
        """Log activation."""
        self.logger.warning(
            "Activated NoOp fallback for embeddings - semantic search disabled",
        )

    def on_deactivate(self) -> None:
        """Log deactivation."""
        self.logger.info("Deactivated NoOp embeddings fallback")


class DirectStorageFallback(FallbackStrategy):
    """Direct storage fallback that bypasses cache."""

    def __init__(self, client) -> None:
        self.client = client
        self.logger = logging.getLogger(f"{__name__}.DirectStorageFallback")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute direct storage operations."""
        if operation == "get_checkpoint":
            # Bypass cache, go directly to storage
            checkpoint_id = kwargs.get("checkpoint_id")
            return self.client.get_checkpoint(checkpoint_id)

        if operation == "store_checkpoint":
            # Store directly without caching
            return self.client.store_checkpoint(*args, **kwargs)

        if operation == "list_checkpoints":
            # List directly from storage
            return self.client.list_checkpoints(*args, **kwargs)

        self.logger.warning(f"Unknown storage operation: {operation}")
        return None

    def is_available(self) -> bool:
        """Available if client is available."""
        return self.client is not None

    def on_activate(self) -> None:
        """Log activation."""
        self.logger.warning("Activated direct storage fallback - cache bypassed")

    def on_deactivate(self) -> None:
        """Log deactivation."""
        self.logger.info("Deactivated direct storage fallback")


class SingleConnectionFallback(FallbackStrategy):
    """Single connection fallback when connection pool fails."""

    def __init__(self, create_client_func: Callable) -> None:
        self.create_client_func = create_client_func
        self._single_client = None
        self.logger = logging.getLogger(f"{__name__}.SingleConnectionFallback")

    def _get_single_client(self):
        """Get or create single client instance."""
        if self._single_client is None:
            self._single_client = self.create_client_func()
        return self._single_client

    def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute operations with single connection."""
        client = self._get_single_client()

        if hasattr(client, operation):
            method = getattr(client, operation)
            return method(*args, **kwargs)
        self.logger.warning(f"Operation {operation} not found on client")
        return None

    def is_available(self) -> bool:
        """Check if we can create a single client."""
        try:
            client = self._get_single_client()
            return client is not None
        except Exception as e:
            self.logger.exception(f"Single client creation failed: {e}")
            return False

    def on_activate(self) -> None:
        """Log activation."""
        self.logger.warning(
            "Activated single connection fallback - connection pool bypassed",
        )

    def on_deactivate(self) -> None:
        """Clean up single client."""
        if self._single_client:
            # Close client if it has close method
            if hasattr(self._single_client, "close"):
                self._single_client.close()
            self._single_client = None
        self.logger.info("Deactivated single connection fallback")


class GracefulDegradation:
    """Main graceful degradation coordinator."""

    def __init__(self, config: DegradationConfig = None) -> None:
        self.config = config or DegradationConfig()
        self.service_level = ServiceLevel.FULL
        self.components: dict[str, ComponentStatus] = {}
        self.fallbacks: dict[str, FallbackStrategy] = {}
        self.active_fallbacks: set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.GracefulDegradation")

        # Threading for health checks
        self._health_check_thread = None
        self._stop_health_checks = False
        self._lock = threading.Lock()

        # Circuit breaker state
        self._circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._circuit_failure_count = 0
        self._circuit_last_failure_time = 0

        # Metrics
        self.metrics = {
            "total_operations": 0,
            "failed_operations": 0,
            "fallback_activations": 0,
            "recovery_attempts": 0,
            "circuit_breaker_trips": 0,
        }

    def start_health_monitoring(self) -> None:
        """Start automatic health monitoring."""
        if self.config.automatic_recovery and not self._health_check_thread:
            self._stop_health_checks = False
            self._health_check_thread = threading.Thread(
                target=self._health_monitor_loop, daemon=True,
            )
            self._health_check_thread.start()
            self.logger.info("Started automatic health monitoring")

    def stop_health_monitoring(self) -> None:
        """Stop automatic health monitoring."""
        self._stop_health_checks = True
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
            self._health_check_thread = None
            self.logger.info("Stopped health monitoring")

    def _health_monitor_loop(self) -> None:
        """Background thread for health monitoring."""
        while not self._stop_health_checks:
            try:
                self._perform_health_checks()
                self._attempt_recovery()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.exception(f"Health monitor error: {e}")
                time.sleep(self.config.health_check_interval)

    def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        with self._lock:
            for name, component in self.components.items():
                try:
                    # Simulate health check - in real implementation would call actual health check
                    is_healthy = component.error_count < self.config.max_error_threshold
                    component.is_healthy = is_healthy
                    component.last_check = time.time()

                    if not is_healthy and name not in self.active_fallbacks:
                        self._try_activate_fallback(name)

                except Exception as e:
                    self.logger.exception(f"Health check failed for {name}: {e}")
                    component.is_healthy = False

    def _attempt_recovery(self) -> None:
        """Attempt to recover failed components."""
        current_time = time.time()

        for name, component in self.components.items():
            if (
                not component.is_healthy
                and component.recovery_attempts < self.config.max_recovery_attempts
                and current_time - component.last_recovery_attempt
                > self.config.recovery_timeout
            ):
                try:
                    # Attempt recovery
                    component.recovery_attempts += 1
                    component.last_recovery_attempt = current_time
                    self.metrics["recovery_attempts"] += 1

                    # Reset error count to test recovery
                    component.error_count = max(0, component.error_count - 2)

                    if component.error_count < self.config.max_error_threshold:
                        component.is_healthy = True
                        self.deactivate_fallback(name)
                        self.logger.info(f"Component {name} recovered")

                except Exception as e:
                    self.logger.exception(f"Recovery attempt failed for {name}: {e}")

    def _try_activate_fallback(self, component_name: str) -> None:
        """Try to activate appropriate fallback for component."""
        fallback_mapping = {
            "qdrant": "InMemoryQdrantFallback",
            "embeddings": "NoOpEmbeddingsFallback",
            "cache": "DirectStorageFallback",
            "connection_pool": "SingleConnectionFallback",
        }

        fallback_name = fallback_mapping.get(component_name)
        if fallback_name:
            self.activate_fallback(fallback_name)

    def register_component(
        self, name: str, health_check_func: Callable | None = None,
    ) -> None:
        """Register a component for health monitoring."""
        self.components[name] = ComponentStatus(name=name)
        self.logger.info(f"Registered component: {name}")

    def register_fallback(self, name: str, strategy: FallbackStrategy) -> None:
        """Register a fallback strategy."""
        self.fallbacks[name] = strategy
        self.logger.info(f"Registered fallback strategy: {name}")

    def check_component_health(self, name: str) -> bool:
        """Check health of a specific component."""
        if name not in self.components:
            return False

        component = self.components[name]
        # Implementation would include actual health check logic
        # For now, simulate based on error count
        is_healthy = component.error_count < self.config.max_error_threshold

        component.is_healthy = is_healthy
        component.last_check = time.time()

        return is_healthy

    def update_service_level(self) -> None:
        """Update overall service level based on component health."""
        healthy_components = sum(1 for c in self.components.values() if c.is_healthy)
        total_components = len(self.components)

        if total_components in (0, healthy_components):
            self.service_level = ServiceLevel.FULL
        elif healthy_components > total_components * 0.5:
            self.service_level = ServiceLevel.DEGRADED
        elif healthy_components > 0:
            self.service_level = ServiceLevel.MINIMAL
        else:
            self.service_level = ServiceLevel.OFFLINE

        self.logger.info(f"Service level updated to: {self.service_level.value}")

    def activate_fallback(self, name: str) -> None:
        """Activate a specific fallback strategy."""
        if name in self.fallbacks and name not in self.active_fallbacks:
            fallback = self.fallbacks[name]
            if fallback.is_available():
                fallback.on_activate()
                self.active_fallbacks.add(name)
                self.metrics["fallback_activations"] += 1
                self.logger.warning(f"Activated fallback: {name}")

    def deactivate_fallback(self, name: str) -> None:
        """Deactivate a specific fallback strategy."""
        if name in self.active_fallbacks:
            fallback = self.fallbacks[name]
            fallback.on_deactivate()
            self.active_fallbacks.remove(name)
            self.logger.info(f"Deactivated fallback: {name}")

    def record_operation(self, success: bool) -> None:
        """Record operation result for metrics and circuit breaker."""
        self.metrics["total_operations"] += 1

        if not success:
            self.metrics["failed_operations"] += 1
            self._circuit_failure_count += 1
            self._circuit_last_failure_time = time.time()

            # Check circuit breaker
            if (
                self._circuit_failure_count >= self.config.circuit_breaker_threshold
                and self._circuit_breaker_state == "CLOSED"
            ):
                self._circuit_breaker_state = "OPEN"
                self.metrics["circuit_breaker_trips"] += 1
                self.logger.error("Circuit breaker OPENED - too many failures")
        else:
            # Reset failure count on success
            self._circuit_failure_count = max(0, self._circuit_failure_count - 1)

            # Try to close circuit breaker
            if (
                self._circuit_breaker_state == "OPEN"
                and time.time() - self._circuit_last_failure_time
                > self.config.recovery_timeout
            ):
                self._circuit_breaker_state = "HALF_OPEN"
                self.logger.info("Circuit breaker HALF_OPEN - testing recovery")
            elif self._circuit_breaker_state == "HALF_OPEN":
                self._circuit_breaker_state = "CLOSED"
                self.logger.info("Circuit breaker CLOSED - service recovered")

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_breaker_state == "OPEN"

    @contextmanager
    def execute_with_fallback(
        self, operation: str, fallback_names: list[str] | None = None,
    ):
        """Context manager for executing operations with fallback support."""
        fallback_names = fallback_names or list(self.fallbacks.keys())

        try:
            yield self
        except Exception as e:
            self.logger.exception(f"Operation {operation} failed: {e}")

            # Try fallbacks in order
            for fallback_name in fallback_names:
                if fallback_name in self.fallbacks:
                    fallback = self.fallbacks[fallback_name]
                    if fallback.is_available():
                        try:
                            result = fallback.execute(operation)
                            self.activate_fallback(fallback_name)
                            return result
                        except Exception as fallback_error:
                            self.logger.exception(
                                f"Fallback {fallback_name} failed: {fallback_error}",
                            )

            # All fallbacks failed
            raise

    def setup_default_fallbacks(self, client=None, create_client_func=None) -> None:
        """Setup default fallback strategies."""
        # Register default fallbacks
        self.register_fallback("InMemoryQdrantFallback", InMemoryQdrantFallback())
        self.register_fallback("NoOpEmbeddingsFallback", NoOpEmbeddingsFallback())

        if client:
            self.register_fallback(
                "DirectStorageFallback", DirectStorageFallback(client),
            )

        if create_client_func:
            self.register_fallback(
                "SingleConnectionFallback", SingleConnectionFallback(create_client_func),
            )

        self.logger.info("Default fallback strategies configured")

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "circuit_breaker_state": self._circuit_breaker_state,
            "active_fallbacks_count": len(self.active_fallbacks),
            "components_healthy": sum(
                1 for c in self.components.values() if c.is_healthy
            ),
            "components_total": len(self.components),
        }


class CheckpointerWithDegradation:
    """Wrapper for UnifiedCheckpointer with graceful degradation support."""

    def __init__(self, checkpointer, degradation: GracefulDegradation = None) -> None:
        self.checkpointer = checkpointer
        self.degradation = degradation or GracefulDegradation()
        self.logger = logging.getLogger(f"{__name__}.CheckpointerWithDegradation")

        # Setup degradation system
        self._setup_degradation()

    def _setup_degradation(self) -> None:
        """Initialize degradation system with checkpointer components."""
        # Register components to monitor
        self.degradation.register_component("qdrant")
        self.degradation.register_component("embeddings")
        self.degradation.register_component("cache")
        self.degradation.register_component("connection_pool")

        # Setup fallbacks with checkpointer context
        self.degradation.setup_default_fallbacks(
            client=getattr(self.checkpointer, "client", None),
            create_client_func=lambda: self.checkpointer._create_single_client()
            if hasattr(self.checkpointer, "_create_single_client")
            else None,
        )

        # Start health monitoring
        self.degradation.start_health_monitoring()

    def _execute_with_monitoring(
        self, operation: str, method: Callable, *args, **kwargs,
    ):
        """Execute operation with health monitoring and fallback support."""
        # Check circuit breaker
        if self.degradation.is_circuit_open():
            msg = f"Circuit breaker is OPEN - operation {operation} blocked"
            raise Exception(msg)

        try:
            # Execute primary operation
            result = method(*args, **kwargs)

            # Record success
            self.degradation.record_operation(success=True)

            # Reset error count on success
            for component in self.degradation.components.values():
                component.error_count = max(0, component.error_count - 1)

            return result

        except Exception as e:
            # Record failure
            self.degradation.record_operation(success=False)

            # Increment error count
            for component in self.degradation.components.values():
                component.error_count += 1
                component.last_error = str(e)

            # Update service level
            self.degradation.update_service_level()

            # Try fallback execution
            with self.degradation.execute_with_fallback(operation):
                # If we reach here, fallback will be attempted
                raise

    async def _execute_with_monitoring_async(
        self, operation: str, method: Callable, *args, **kwargs,
    ):
        """Async version of execute with monitoring."""
        # Check circuit breaker
        if self.degradation.is_circuit_open():
            msg = f"Circuit breaker is OPEN - operation {operation} blocked"
            raise Exception(msg)

        try:
            # Execute primary operation
            result = await method(*args, **kwargs)

            # Record success
            self.degradation.record_operation(success=True)

            # Reset error count on success
            for component in self.degradation.components.values():
                component.error_count = max(0, component.error_count - 1)

            return result

        except Exception as e:
            # Record failure
            self.degradation.record_operation(success=False)

            # Increment error count
            for component in self.degradation.components.values():
                component.error_count += 1
                component.last_error = str(e)

            # Update service level
            self.degradation.update_service_level()

            # Re-raise for now - async fallback would need more complex implementation
            raise

    def put(self, config, checkpoint, metadata, **kwargs):
        """Put with degradation support."""
        return self._execute_with_monitoring(
            "put", self.checkpointer.put, config, checkpoint, metadata, **kwargs,
        )

    def get_tuple(self, config, **kwargs):
        """Get tuple with degradation support."""
        return self._execute_with_monitoring(
            "get_tuple", self.checkpointer.get_tuple, config, **kwargs,
        )

    def list(self, config, **kwargs):
        """List with degradation support."""
        return self._execute_with_monitoring(
            "list", self.checkpointer.list, config, **kwargs,
        )

    def put_writes(self, config, writes, **kwargs):
        """Put writes with degradation support."""
        return self._execute_with_monitoring(
            "put_writes", self.checkpointer.put_writes, config, writes, **kwargs,
        )

    # Add async methods
    async def aput(self, config, checkpoint, metadata, **kwargs):
        """Async put with degradation support."""
        return await self._execute_with_monitoring_async(
            "aput", self.checkpointer.aput, config, checkpoint, metadata, **kwargs,
        )

    async def aget_tuple(self, config, **kwargs):
        """Async get tuple with degradation support."""
        return await self._execute_with_monitoring_async(
            "aget_tuple", self.checkpointer.aget_tuple, config, **kwargs,
        )

    async def alist(self, config, **kwargs):
        """Async list with degradation support."""
        return await self._execute_with_monitoring_async(
            "alist", self.checkpointer.alist, config, **kwargs,
        )

    async def aput_writes(self, config, writes, **kwargs):
        """Async put writes with degradation support."""
        return await self._execute_with_monitoring_async(
            "aput_writes", self.checkpointer.aput_writes, config, writes, **kwargs,
        )

    def get_service_status(self) -> dict[str, Any]:
        """Get current service status and degradation info."""
        return {
            "service_level": self.degradation.service_level.value,
            "components": {
                name: {
                    "healthy": status.is_healthy,
                    "error_count": status.error_count,
                    "last_error": status.last_error,
                    "last_check": status.last_check,
                    "recovery_attempts": status.recovery_attempts,
                }
                for name, status in self.degradation.components.items()
            },
            "active_fallbacks": list(self.degradation.active_fallbacks),
            "metrics": self.degradation.get_metrics(),
            "circuit_breaker": self.degradation._circuit_breaker_state,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.degradation.stop_health_monitoring()
