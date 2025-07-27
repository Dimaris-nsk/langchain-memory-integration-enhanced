"""Tests for circuit breaker functionality.

This module contains comprehensive tests for the CircuitBreaker class,
covering all states, transitions, and edge cases.
"""

import time
from typing import Never
from unittest.mock import patch

import pytest
from unified_checkpointer.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    circuit_breaker,
    get_circuit_breaker,
)


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    def test_initial_state(self) -> None:
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    def test_custom_config(self) -> None:
        """Test circuit breaker with custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window=30.0,
            success_threshold=2,
            recovery_timeout=15.0,
            name="test_breaker",
        )
        cb = CircuitBreaker(config)
        assert cb.config.failure_threshold == 3
        assert cb.config.failure_window == 30.0
        assert cb.config.success_threshold == 2
        assert cb.config.recovery_timeout == 15.0
        assert cb.config.name == "test_breaker"

    def test_successful_call(self) -> None:
        """Test successful function call through circuit breaker."""
        cb = CircuitBreaker()

        def success_func() -> str:
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb._stats.success_count == 1
        assert cb._stats.failure_count == 0
        assert cb._stats.total_calls == 1
        assert cb.state == CircuitState.CLOSED

    def test_failure_call(self) -> None:
        """Test failed function call through circuit breaker."""
        cb = CircuitBreaker()

        def failure_func() -> Never:
            msg = "Test failure"
            raise Exception(msg)

        with pytest.raises(Exception, match="Test failure"):
            cb.call(failure_func)

        assert cb._stats.failure_count == 1
        assert cb._stats.success_count == 0
        assert cb._stats.total_calls == 1
        assert cb._stats.total_failures == 1
        assert cb.state == CircuitState.CLOSED  # Still closed after 1 failure

    def test_circuit_opens_after_threshold(self) -> None:
        """Test circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3, failure_window=60.0)
        cb = CircuitBreaker(config)

        def failure_func() -> Never:
            msg = "Test failure"
            raise Exception(msg)

        # Fail 3 times to reach threshold
        for _i in range(3):
            with pytest.raises(Exception):
                cb.call(failure_func)

        # Circuit should be open now
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
        assert cb._stats.circuit_opened_count == 1

        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "should not execute")

    def test_failure_window_expiry(self) -> None:
        """Test failures outside window don't count towards threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window=0.1,  # 100ms window
        )
        cb = CircuitBreaker(config)

        def failure_func() -> Never:
            msg = "Test failure"
            raise Exception(msg)

        # First failure
        with pytest.raises(Exception):
            cb.call(failure_func)

        # Wait for window to expire
        time.sleep(0.2)

        # This failure should not count with the first one
        with pytest.raises(Exception):
            cb.call(failure_func)

        # Add one more failure
        with pytest.raises(Exception):
            cb.call(failure_func)

        # Circuit should still be closed (only 2 failures in window)
        assert cb.state == CircuitState.CLOSED



class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery and HALF_OPEN state."""

    def test_transition_to_half_open_after_timeout(self) -> None:
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for quick test
        )
        cb = CircuitBreaker(config)

        def failure_func() -> Never:
            msg = "Test failure"
            raise Exception(msg)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failure_func)

        assert cb.state == CircuitState.OPEN

        # Try to call immediately - should fail
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "test")

        # Wait for recovery timeout
        time.sleep(0.15)

        # Now it should be in HALF_OPEN and allow one attempt
        # Use a success function to test the transition
        result = cb.call(lambda: "success")
        assert result == "success"

        # After success in HALF_OPEN, it should need more successes
        assert cb.state == CircuitState.HALF_OPEN

    def test_successful_recovery_from_half_open(self) -> None:
        """Test successful recovery: HALF_OPEN -> CLOSED."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,  # Need 2 successes to close
            recovery_timeout=0.1,
        )
        cb = CircuitBreaker(config)

        def failure_func() -> Never:
            msg = "Test failure"
            raise Exception(msg)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failure_func)

        # Wait for recovery timeout
        time.sleep(0.15)

        # Make successful calls to recover
        for i in range(2):
            result = cb.call(lambda: f"success_{i}")
            assert result == f"success_{i}"

        # Should be back to CLOSED state
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed


    def test_failed_recovery_from_half_open(self) -> None:
        """Test failed recovery: HALF_OPEN -> OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
        )
        cb = CircuitBreaker(config)

        def failure_func() -> Never:
            msg = "Still failing"
            raise Exception(msg)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failure_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Try to recover but fail - should go back to OPEN
        with pytest.raises(Exception):
            cb.call(failure_func)

        # Should be back to OPEN state
        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    def test_mixed_results_in_half_open(self) -> None:
        """Test mixed success/failure in HALF_OPEN state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=3,  # Need 3 consecutive successes
            recovery_timeout=0.1,
        )
        cb = CircuitBreaker(config)

        def failure_func() -> Never:
            msg = "Failure"
            raise Exception(msg)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failure_func)

        # Wait for recovery timeout
        time.sleep(0.15)

        # Success, then failure - should reset and go to OPEN
        cb.call(lambda: "success1")
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(Exception):
            cb.call(failure_func)

        # Should be back to OPEN after failure in HALF_OPEN
        assert cb.state == CircuitState.OPEN


    @patch("time.time")
    def test_recovery_timeout_with_mock_time(self, mock_time) -> None:
        """Test recovery timeout using mocked time for stability."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=30.0,  # 30 seconds
        )
        cb = CircuitBreaker(config)

        # Set initial time
        mock_time.return_value = 1000.0

        def failure_func() -> Never:
            msg = "Failure"
            raise Exception(msg)

        # Open the circuit at t=1000
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failure_func)

        assert cb.state == CircuitState.OPEN

        # Try at t=1010 (10s later) - still OPEN
        mock_time.return_value = 1010.0
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "test")

        # Try at t=1030 (30s later) - should transition to HALF_OPEN
        mock_time.return_value = 1030.0
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN



class TestCircuitBreakerAsync:
    """Test async circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_async_successful_call(self) -> None:
        """Test successful async function call through circuit breaker."""
        cb = CircuitBreaker()

        async def async_success() -> str:
            return "async_success"

        result = await cb.call_async(async_success)
        assert result == "async_success"
        assert cb._stats.success_count == 1
        assert cb._stats.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_failed_call(self) -> None:
        """Test failed async function call through circuit breaker."""
        cb = CircuitBreaker()

        async def async_failure() -> Never:
            msg = "Async failure"
            raise Exception(msg)

        with pytest.raises(Exception, match="Async failure"):
            await cb.call_async(async_failure)

        assert cb._stats.failure_count == 1
        assert cb._stats.success_count == 0
        assert cb.state == CircuitState.CLOSED


    @pytest.mark.asyncio
    async def test_async_circuit_opens_on_failure(self) -> None:
        """Test async circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config)

        async def async_failure() -> Never:
            msg = "Async failure"
            raise Exception(msg)

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call_async(async_failure)

        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await cb.call_async(async_failure)

    @pytest.mark.asyncio
    async def test_async_bypass_exceptions(self) -> None:
        """Test async call with bypass exceptions."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            expected_exceptions=(ValueError,),
            bypass_exceptions=(TypeError,),
        )
        cb = CircuitBreaker(config)

        async def raise_bypass_exception() -> Never:
            msg = "Should bypass circuit breaker"
            raise TypeError(msg)

        # Bypass exception should not affect circuit state
        with pytest.raises(TypeError):
            await cb.call_async(raise_bypass_exception)

        assert cb._stats.failure_count == 0
        assert cb.state == CircuitState.CLOSED



class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""

    def test_decorator_usage(self) -> None:
        """Test using circuit breaker as decorator."""
        cb = CircuitBreaker()

        @cb
        def decorated_function(x):
            return x * 2

        result = decorated_function(5)
        assert result == 10
        assert cb._stats.success_count == 1

    def test_decorator_with_failure(self) -> None:
        """Test decorator with failing function."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config)

        @cb
        def failing_function() -> Never:
            msg = "Decorated failure"
            raise ValueError(msg)

        # First failure
        with pytest.raises(ValueError):
            failing_function()

        assert cb._stats.failure_count == 1

        # Second failure opens circuit
        with pytest.raises(ValueError):
            failing_function()

        assert cb.state == CircuitState.OPEN

        # Third call blocked by open circuit
        with pytest.raises(CircuitOpenError):
            failing_function()


    def test_global_decorator(self) -> None:
        """Test using global circuit_breaker decorator."""

        @circuit_breaker(name="test_api", failure_threshold=2)
        def api_call() -> str:
            return "success"

        # Should work normally
        result = api_call()
        assert result == "success"

        # Get the breaker to check stats
        breaker = get_circuit_breaker("test_api")
        assert breaker._stats.success_count == 1
        assert breaker.state == CircuitState.CLOSED

    def test_global_decorator_with_params(self) -> None:
        """Test global decorator with custom parameters."""

        @circuit_breaker(
            name="custom_api",
            failure_threshold=1,
            recovery_timeout=10.0,
            expected_exceptions=(ValueError, TypeError),
        )
        def custom_api() -> Never:
            msg = "API error"
            raise ValueError(msg)

        # First call fails and opens circuit
        with pytest.raises(ValueError):
            custom_api()

        # Second call blocked
        with pytest.raises(CircuitOpenError):
            custom_api()

        breaker = get_circuit_breaker("custom_api")
        assert breaker.state == CircuitState.OPEN
        assert breaker.config.recovery_timeout == 10.0



class TestCircuitBreakerUtilities:
    """Test utility functions and edge cases."""

    def test_get_stats(self) -> None:
        """Test statistics retrieval."""
        cb = CircuitBreaker("stats_test")

        # Initial stats
        stats = cb.get_stats()
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 0
        assert stats["consecutive_failures"] == 0
        assert stats["state"] == "CLOSED"

        # After success
        result = cb.call(lambda: "ok")
        assert result == "ok"

        stats = cb.get_stats()
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 0

        # After failure
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("test")))

        stats = cb.get_stats()
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 1

    def test_reset(self) -> None:
        """Test manual reset functionality."""
        cb = CircuitBreaker("reset_test", failure_threshold=1)

        # Open circuit
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("test")))

        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._stats.failure_count == 0
        assert cb._stats.success_count == 0
        assert cb._stats.consecutive_failures == 0

    def test_force_open(self) -> None:
        """Test forcing circuit open."""
        cb = CircuitBreaker("force_test")

        # Normal state
        assert cb.state == CircuitState.CLOSED

        # Force open
        cb.force_open()
        assert cb.state == CircuitState.OPEN

        # Should block calls
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "test")

    def test_state_override_context(self) -> None:
        """Test state override context manager."""
        cb = CircuitBreaker("override_test", failure_threshold=1)

        # Open circuit
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("test")))

        assert cb.state == CircuitState.OPEN

        # Override to closed state
        with cb.override_state(CircuitState.CLOSED):
            # Should allow calls
            result = cb.call(lambda: "success")
            assert result == "success"

        # Back to open after context
        assert cb.state == CircuitState.OPEN