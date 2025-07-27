"""
Monitoring and metrics module for UnifiedCheckpointer.

Provides comprehensive observability through OpenTelemetry metrics
with Prometheus export capability.
"""

import builtins
import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, TypeVar

# OpenTelemetry imports
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from prometheus_client import start_http_server

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logging.warning(
        "OpenTelemetry not installed. Metrics collection disabled. "
        "Install with: pip install opentelemetry-api opentelemetry-sdk "
        "opentelemetry-exporter-prometheus prometheus-client",
    )

T = TypeVar("T")


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    enabled: bool = True
    service_name: str = "unified-checkpointer"
    service_version: str = "1.0.0"
    prometheus_port: int = 8080
    prometheus_addr: str = "0.0.0.0"
    namespace: str = "langchain_memory"

    # Metric buckets
    size_buckets: tuple = (
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
    )  # bytes
    duration_buckets: tuple = (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    )  # seconds


class MetricsCollector:
    """Central metrics collection for UnifiedCheckpointer."""

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self.config = config or MetricsConfig()
        self._meter = None
        self._metrics: dict[str, Any] = {}

        if METRICS_ENABLED and self.config.enabled:
            self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize OpenTelemetry metrics provider and instruments."""
        try:
            # Create resource
            resource = Resource.create(
                {
                    "service.name": self.config.service_name,
                    "service.version": self.config.service_version,
                    "service.namespace": self.config.namespace,
                },
            )

            # Start Prometheus HTTP server
            start_http_server(
                port=self.config.prometheus_port, addr=self.config.prometheus_addr,
            )
            logging.info(
                f"Prometheus metrics server started on "
                f"{self.config.prometheus_addr}:{self.config.prometheus_port}",
            )

            # Create MeterProvider with Prometheus exporter
            reader = PrometheusMetricReader()
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)

            # Get meter
            self._meter = metrics.get_meter(
                self.config.service_name, self.config.service_version,
            )

            # Create metrics instruments
            self._create_instruments()

        except Exception as e:
            logging.exception(f"Failed to initialize metrics: {e}")
            self.config.enabled = False

    def _create_instruments(self) -> None:
        """Create all metric instruments."""
        if not self._meter:
            return

        # Counters
        self._metrics["checkpoint_counter"] = self._meter.create_counter(
            name=f"{self.config.namespace}_checkpoints_total",
            description="Total number of checkpoints saved",
            unit="1",
        )

        self._metrics["search_counter"] = self._meter.create_counter(
            name=f"{self.config.namespace}_searches_total",
            description="Total number of searches performed",
            unit="1",
        )

        self._metrics["cache_hits"] = self._meter.create_counter(
            name=f"{self.config.namespace}_cache_hits_total",
            description="Number of cache hits",
            unit="1",
        )

        self._metrics["cache_misses"] = self._meter.create_counter(
            name=f"{self.config.namespace}_cache_misses_total",
            description="Number of cache misses",
            unit="1",
        )

        self._metrics["error_counter"] = self._meter.create_counter(
            name=f"{self.config.namespace}_errors_total",
            description="Total number of errors",
            unit="1",
        )

        self._metrics["retry_counter"] = self._meter.create_counter(
            name=f"{self.config.namespace}_retries_total",
            description="Total number of retry attempts",
            unit="1",
        )

        # Histograms
        self._metrics["checkpoint_size"] = self._meter.create_histogram(
            name=f"{self.config.namespace}_checkpoint_size_bytes",
            description="Size of checkpoints in bytes",
            unit="By",
        )

        self._metrics["operation_duration"] = self._meter.create_histogram(
            name=f"{self.config.namespace}_operation_duration_seconds",
            description="Duration of checkpoint operations",
            unit="s",
        )

        self._metrics["batch_size"] = self._meter.create_histogram(
            name=f"{self.config.namespace}_batch_size",
            description="Size of batch operations",
            unit="1",
        )

        # Gauges (UpDownCounter in OTel)
        self._metrics["active_threads"] = self._meter.create_up_down_counter(
            name=f"{self.config.namespace}_active_threads",
            description="Number of active conversation threads",
            unit="1",
        )

        self._metrics["cache_size"] = self._meter.create_up_down_counter(
            name=f"{self.config.namespace}_cache_size_items",
            description="Number of items in cache",
            unit="1",
        )

        self._metrics["pool_connections"] = self._meter.create_up_down_counter(
            name=f"{self.config.namespace}_pool_connections",
            description="Number of active connection pool connections",
            unit="1",
        )

        # Observable gauges for system metrics
        self._metrics["memory_usage"] = self._meter.create_observable_gauge(
            name=f"{self.config.namespace}_memory_usage_bytes",
            callbacks=[self._get_memory_usage],
            description="Process memory usage in bytes",
            unit="By",
        )

    def _get_memory_usage(self, options) -> None:
        """Callback for memory usage metric."""
        try:
            import psutil

            process = psutil.Process()
            yield metrics.Observation(process.memory_info().rss, {"type": "rss"})
        except ImportError:
            pass

    # Decorator for timing operations
    def time_operation(self, operation: str):
        """Decorator to time function execution."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                if not self.config.enabled or not self._meter:
                    return func(*args, **kwargs)

                start_time = time.time()
                labels = {"operation": operation}

                try:
                    result = func(*args, **kwargs)
                    labels["status"] = "success"
                    return result
                except Exception as e:
                    labels["status"] = "error"
                    labels["error_type"] = type(e).__name__
                    self.record_error(operation, type(e).__name__)
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_duration(operation, duration, labels)

            return wrapper

        return decorator

    # Context manager for timing operations
    @contextmanager
    def measure_operation(self, operation: str, **labels):
        """Context manager to measure operation duration."""
        if not self.config.enabled or not self._meter:
            yield
            return

        start_time = time.time()
        labels["operation"] = operation

        try:
            yield
            labels["status"] = "success"
        except Exception as e:
            labels["status"] = "error"
            labels["error_type"] = type(e).__name__
            self.record_error(operation, type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            self.record_duration(operation, duration, labels)

    # Recording methods
    def record_checkpoint_saved(self, thread_id: str, size: int, **labels) -> None:
        """Record a checkpoint save operation."""
        if not self.config.enabled or not self._meter:
            return

        labels["thread_id"] = thread_id
        self._metrics["checkpoint_counter"].add(1, labels)
        self._metrics["checkpoint_size"].record(size, labels)

    def record_search(self, search_type: str, results_count: int, **labels) -> None:
        """Record a search operation."""
        if not self.config.enabled or not self._meter:
            return

        labels["search_type"] = search_type
        labels["has_results"] = str(results_count > 0)
        self._metrics["search_counter"].add(1, labels)

    def record_cache_hit(self, cache_type: str = "checkpoint") -> None:
        """Record a cache hit."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["cache_hits"].add(1, {"cache_type": cache_type})

    def record_cache_miss(self, cache_type: str = "checkpoint") -> None:
        """Record a cache miss."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["cache_misses"].add(1, {"cache_type": cache_type})

    def record_error(self, operation: str, error_type: str) -> None:
        """Record an error occurrence."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["error_counter"].add(
            1, {"operation": operation, "error_type": error_type},
        )

    def record_retry(self, operation: str, attempt: int) -> None:
        """Record a retry attempt."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["retry_counter"].add(
            1, {"operation": operation, "attempt": str(attempt)},
        )

    def record_duration(self, operation: str, duration: float, labels: dict[str, str]) -> None:
        """Record operation duration."""
        if not self.config.enabled or not self._meter:
            return

        labels["operation"] = operation
        self._metrics["operation_duration"].record(duration, labels)

    def record_batch_size(self, operation: str, size: int) -> None:
        """Record batch operation size."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["batch_size"].record(size, {"operation": operation})

    def update_active_threads(self, delta: int) -> None:
        """Update active threads count."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["active_threads"].add(delta)

    def update_cache_size(self, delta: int) -> None:
        """Update cache size."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["cache_size"].add(delta)

    def update_pool_connections(self, delta: int) -> None:
        """Update connection pool size."""
        if not self.config.enabled or not self._meter:
            return

        self._metrics["pool_connections"].add(delta)

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (for reporting)."""
        # Note: This is for local calculation only
        # Actual rate should be calculated in Prometheus/Grafana
        return 0.0  # Placeholder


# Global metrics instance
_metrics_collector: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def configure_metrics(config: MetricsConfig) -> MetricsCollector:
    """Configure and return metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(config)
    return _metrics_collector


# Convenience decorators for easy integration
def track_operation(operation: str):
    """Decorator to track operation metrics."""
    return get_metrics().time_operation(operation)


def track_batch_operation(operation: str):
    """Decorator for batch operations that extracts batch size."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            metrics = get_metrics()

            # Try to extract batch size from arguments
            batch_size = None
            if args and hasattr(args[0], "__len__"):
                with suppress(builtins.BaseException):
                    batch_size = len(args[0])
            elif "configs" in kwargs and hasattr(kwargs["configs"], "__len__"):
                batch_size = len(kwargs["configs"])
            elif "checkpoints" in kwargs and hasattr(kwargs["checkpoints"], "__len__"):
                batch_size = len(kwargs["checkpoints"])

            if batch_size is not None:
                metrics.record_batch_size(operation, batch_size)

            # Use the standard timing decorator
            return metrics.time_operation(operation)(func)(*args, **kwargs)

        return wrapper

    return decorator


# Integration helpers
class MetricsIntegration:
    """Helper class for integrating metrics into UnifiedCheckpointer."""

    @staticmethod
    def on_checkpoint_save(thread_id: str, checkpoint_data: bytes) -> None:
        """Call after saving a checkpoint."""
        metrics = get_metrics()
        metrics.record_checkpoint_saved(thread_id, len(checkpoint_data))
        metrics.update_active_threads(1)

    @staticmethod
    def on_checkpoint_delete(thread_id: str) -> None:
        """Call after deleting a checkpoint."""
        metrics = get_metrics()
        metrics.update_active_threads(-1)

    @staticmethod
    def on_search(search_type: str, query: str, results: list) -> None:
        """Call after performing a search."""
        metrics = get_metrics()
        metrics.record_search(search_type, len(results), query_length=len(query))

    @staticmethod
    def on_cache_check(hit: bool, cache_type: str = "checkpoint") -> None:
        """Call after checking cache."""
        metrics = get_metrics()
        if hit:
            metrics.record_cache_hit(cache_type)
        else:
            metrics.record_cache_miss(cache_type)

    @staticmethod
    def on_retry(operation: str, attempt: int) -> None:
        """Call when retrying an operation."""
        metrics = get_metrics()
        metrics.record_retry(operation, attempt)

    @staticmethod
    def on_connection_acquired() -> None:
        """Call when acquiring a connection from pool."""
        metrics = get_metrics()
        metrics.update_pool_connections(1)

    @staticmethod
    def on_connection_released() -> None:
        """Call when releasing a connection to pool."""
        metrics = get_metrics()
        metrics.update_pool_connections(-1)


# Example usage in UnifiedCheckpointer:
"""
from .metrics import track_operation, MetricsIntegration

class UnifiedCheckpointer:
    @track_operation("put_checkpoint")
    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata) -> RunnableConfig:
        # ... existing implementation ...

        # Record metrics
        MetricsIntegration.on_checkpoint_save(
            thread_id=config["configurable"]["thread_id"],
            checkpoint_data=serialized_data
        )

        return config
"""