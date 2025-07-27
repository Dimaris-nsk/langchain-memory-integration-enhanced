"""
Circuit breaker implementation for fault tolerance.

Protects against cascading failures by temporarily blocking calls
to failing services.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold before opening circuit
    failure_threshold: int = 5
    # Time window for counting failures (seconds)
    failure_window: float = 60.0
    # Success threshold in HALF_OPEN before closing
    success_threshold: int = 3
    # Time to wait before attempting recovery (seconds)
    recovery_timeout: float = 30.0
    # Expected exceptions that trigger the breaker
    expected_exceptions: tuple = (Exception,)
    # Exceptions that should bypass the breaker
    bypass_exceptions: tuple = ()
    # Name for logging
    name: str = "circuit_breaker"


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    circuit_opened_count: int = 0
    last_state_change: datetime | None = None


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.RLock()
        self._failure_timestamps: list = []
        self._half_open_start: float | None = None

        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_recovery_timeout()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def _check_recovery_timeout(self) -> None:
        """Check if recovery timeout has passed and transition to HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._stats.last_failure_time:
            elapsed = time.time() - self._stats.last_failure_time
            if elapsed >= self.config.recovery_timeout:
                self._transition_to_half_open()

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        with self._lock:
            self.logger.info(f"Circuit {self.config.name} transitioning to CLOSED")
            self._state = CircuitState.CLOSED
            self._stats.consecutive_failures = 0
            self._stats.last_state_change = datetime.now()
            self._failure_timestamps.clear()

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        with self._lock:
            self.logger.warning(f"Circuit {self.config.name} transitioning to OPEN")
            self._state = CircuitState.OPEN
            self._stats.consecutive_successes = 0
            self._stats.circuit_opened_count += 1
            self._stats.last_state_change = datetime.now()

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        with self._lock:
            self.logger.info(f"Circuit {self.config.name} transitioning to HALF_OPEN")
            self._state = CircuitState.HALF_OPEN
            self._stats.consecutive_successes = 0
            self._stats.last_state_change = datetime.now()
            self._half_open_start = time.time()

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.success_count += 1
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()
            self._stats.total_calls += 1

            # Handle state transitions
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            current_time = time.time()
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = current_time
            self._stats.total_calls += 1

            # Add to failure timestamps
            self._failure_timestamps.append(current_time)

            # Remove old timestamps outside the failure window
            cutoff_time = current_time - self.config.failure_window
            self._failure_timestamps = [
                ts for ts in self._failure_timestamps if ts > cutoff_time
            ]

            # Check if we should open the circuit
            if self._state == CircuitState.CLOSED:
                if len(self._failure_timestamps) >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in HALF_OPEN reopens the circuit
                self._transition_to_open()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.

        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            msg = (
                f"Circuit {self.config.name} is OPEN. "
                f"Service unavailable for {self.config.recovery_timeout}s"
            )
            raise CircuitOpenError(
                msg,
            )

        try:
            # Execute the function
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.config.bypass_exceptions:
            # These exceptions bypass the circuit breaker
            raise

        except self.config.expected_exceptions:
            # These exceptions trigger the circuit breaker
            self._record_failure()
            raise

        except Exception:
            # Unexpected exceptions also trigger the breaker
            self._record_failure()
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator usage of circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)

        return wrapper

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Async version of call."""
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            msg = (
                f"Circuit {self.config.name} is OPEN. "
                f"Service unavailable for {self.config.recovery_timeout}s"
            )
            raise CircuitOpenError(
                msg,
            )

        try:
            # Execute the async function
            result = await func(*args, **kwargs)
            self._record_success()
            return result

        except self.config.bypass_exceptions:
            raise

        except self.config.expected_exceptions:
            self._record_failure()
            raise

        except Exception:
            self._record_failure()
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._stats.failure_count,
                "success_count": self._stats.success_count,
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "total_calls": self._stats.total_calls,
                "total_failures": self._stats.total_failures,
                "total_successes": self._stats.total_successes,
                "circuit_opened_count": self._stats.circuit_opened_count,
                "last_failure_time": self._stats.last_failure_time,
                "last_success_time": self._stats.last_success_time,
                "last_state_change": self._stats.last_state_change.isoformat()
                if self._stats.last_state_change
                else None,
                "failure_rate": self._stats.total_failures / self._stats.total_calls
                if self._stats.total_calls > 0
                else 0.0,
            }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._failure_timestamps.clear()
            self._half_open_start = None
            self.logger.info(f"Circuit {self.config.name} has been reset")

    @contextmanager
    def override_state(self, state: CircuitState):
        """Temporarily override circuit state (for testing)."""
        original_state = self._state
        try:
            self._state = state
            yield
        finally:
            self._state = original_state


class CircuitOpenError(Exception):
    """Raised when circuit is open and blocking calls."""


# Global registry for circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig(name=name)
        _circuit_breakers[name] = CircuitBreaker(config)
    return _circuit_breakers[name]


def circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    expected_exceptions: tuple = (Exception,),
):
    """
    Decorator to protect functions with circuit breaker.

    Usage:
        @circuit_breaker(name="external_api", failure_threshold=3)
        def call_external_api():
            ...
    """
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exceptions=expected_exceptions,
    )

    breaker = get_circuit_breaker(name, config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return breaker(func)

    return decorator


# Example integration with UnifiedCheckpointer:
"""
from .circuit_breaker import circuit_breaker, CircuitOpenError

class UnifiedCheckpointer:
    @circuit_breaker(name="qdrant_put", failure_threshold=3, recovery_timeout=60)
    def _store_to_qdrant(self, data: dict):
        # Actual Qdrant storage logic
        pass

    def put(self, config, checkpoint, metadata):
        try:
            self._store_to_qdrant(data)
        except CircuitOpenError:
            # Handle circuit open - maybe use fallback
            logger.warning("Qdrant circuit open, using fallback")
            self._fallback_storage(data)
"""
