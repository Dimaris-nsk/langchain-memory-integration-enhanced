"""Tests for error handling and edge cases in UnifiedCheckpointer.

This file contains tests for scenarios not covered in the main test file,
focusing on error handling, edge cases, and failure scenarios.
"""

import logging
from unittest.mock import AsyncMock, patch

from unified_checkpointer import UnifiedCheckpointer, UnifiedCheckpointerConfig


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_pool_initialization_failure(self, caplog) -> None:
        """Test fallback to direct client when pool initialization fails."""
        # Create config with pool enabled
        config = UnifiedCheckpointerConfig(
            unified_memory_url="http://localhost:8000",
            collection_name="test_checkpoints",
            pool_enabled=True,
        )

        # Create checkpointer
        checkpointer = UnifiedCheckpointer(config=config)

        # Mock the pool to raise exception during initialization
        with patch.object(checkpointer._pool, "initialize", new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = Exception("Pool initialization failed")

            # Set up logging capture
            with caplog.at_level(logging.INFO):
                # Call _ensure_pool_initialized directly
                checkpointer._ensure_pool_initialized()

            # Verify pool initialization was attempted
            mock_init.assert_called_once()

            # Verify pool is set to None after failure
            assert checkpointer._pool is None

            # Verify client was created as fallback
            assert checkpointer._client is not None

            # Verify log messages
            assert "Failed to initialize connection pool" in caplog.text
            assert "Falling back to direct client connection" in caplog.text
