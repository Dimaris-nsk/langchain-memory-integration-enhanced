"""Retry logic with exponential backoff for production stability.

Provides decorators and utilities for automatic retry with configurable
backoff strategies to handle transient failures gracefully.
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from unified_checkpointer.exceptions import UnifiedCheckpointerError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(UnifiedCheckpointerError):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self, message: str, attempts: int, last_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: set[type[Exception]] | None = None,
    ) -> None:
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Set of exceptions to retry on
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or {
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        }

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay,
        )

        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(
            isinstance(exception, exc_type) for exc_type in self.retryable_exceptions
        )


def retry(
    config: RetryConfig | None = None,
    *,
    max_attempts: int | None = None,
    exceptions: set[type[Exception]] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        config: RetryConfig instance
        max_attempts: Override max attempts
        exceptions: Override retryable exceptions

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    if max_attempts is not None:
        config.max_attempts = max_attempts

    if exceptions is not None:
        config.retryable_exceptions = exceptions

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e) or attempt == config.max_attempts:
                        logger.warning(
                            f"Retry exhausted for {func.__name__} after {attempt} attempts: {e}",
                        )
                        msg = f"Failed after {attempt} attempts"
                        raise RetryError(
                            msg,
                            attempts=attempt,
                            last_exception=e,
                        )

                    delay = config.calculate_delay(attempt)
                    logger.info(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}",
                    )
                    time.sleep(delay)

            # Should never reach here
            msg = f"Failed after {config.max_attempts} attempts"
            raise RetryError(
                msg,
                attempts=config.max_attempts,
                last_exception=last_exception,
            )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e) or attempt == config.max_attempts:
                        logger.warning(
                            f"Retry exhausted for {func.__name__} after {attempt} attempts: {e}",
                        )
                        msg = f"Failed after {attempt} attempts"
                        raise RetryError(
                            msg,
                            attempts=attempt,
                            last_exception=e,
                        )

                    delay = config.calculate_delay(attempt)
                    logger.info(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}",
                    )
                    await asyncio.sleep(delay)

            # Should never reach here
            msg = f"Failed after {config.max_attempts} attempts"
            raise RetryError(
                msg,
                attempts=config.max_attempts,
                last_exception=last_exception,
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Pre-configured retry decorators for common scenarios
retry_on_network_errors = retry(
    RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        retryable_exceptions={
            ConnectionError,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
            asyncio.TimeoutError,
        },
    ),
)

retry_on_qdrant_errors = retry(
    RetryConfig(
        max_attempts=5,
        initial_delay=0.5,
        max_delay=30.0,
        retryable_exceptions={
            ConnectionError,
            TimeoutError,
            # Add Qdrant-specific exceptions here
        },
    ),
)


class RetryableClient:
    """Wrapper to add retry logic to any client."""

    def __init__(self, client: Any, retry_config: RetryConfig | None = None) -> None:
        """
        Wrap a client with automatic retry.

        Args:
            client: The client to wrap
            retry_config: Retry configuration
        """
        self._client = client
        self._retry_config = retry_config or RetryConfig()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to wrapped client with retry."""
        attr = getattr(self._client, name)

        if callable(attr):
            # Wrap callable attributes with retry
            return retry(self._retry_config)(attr)

        return attr


# Example usage in checkpointer
def add_retry_to_checkpointer(checkpointer: Any) -> Any:
    """
    Add retry logic to checkpointer methods.

    Args:
        checkpointer: The checkpointer instance

    Returns:
        Checkpointer with retry-enabled methods
    """
    # Decorate critical methods
    checkpointer.put = retry_on_qdrant_errors(checkpointer.put)
    checkpointer.get_tuple = retry_on_qdrant_errors(checkpointer.get_tuple)
    checkpointer.aput = retry_on_qdrant_errors(checkpointer.aput)
    checkpointer.aget_tuple = retry_on_qdrant_errors(checkpointer.aget_tuple)

    return checkpointer
