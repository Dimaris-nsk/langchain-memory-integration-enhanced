"""Tests for UnifiedMemoryClient."""

import pytest


class TestUnifiedMemoryClient:
    """Test UnifiedMemoryClient functionality."""

    @pytest.mark.asyncio
    async def test_store_checkpoint(self, client, sample_checkpoint) -> None:
        """Test storing a checkpoint."""
        # Store checkpoint
        checkpoint_id = await client.store_checkpoint(sample_checkpoint)

        # Verify ID was returned
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)

    @pytest.mark.asyncio
    async def test_get_checkpoint(self, client, sample_checkpoint) -> None:
        """Test retrieving a checkpoint."""
        # Store checkpoint first
        checkpoint_id = await client.store_checkpoint(sample_checkpoint)

        # Retrieve it
        retrieved = await client.get_checkpoint(checkpoint_id)

        # Verify data
        assert retrieved is not None
        assert retrieved["thread_id"] == sample_checkpoint["thread_id"]
        assert retrieved["checkpoint"] == sample_checkpoint["checkpoint"]
