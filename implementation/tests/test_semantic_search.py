"""Tests for semantic search functionality."""

from unittest.mock import patch

import pytest
from unified_checkpointer import UnifiedCheckpointer
from unified_checkpointer.config import UnifiedCheckpointerConfig


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    @pytest.fixture
    def checkpointer(self):
        """Create a checkpointer instance."""
        config = UnifiedCheckpointerConfig(
            collection_name="test_checkpoints",
            unified_memory_url=":memory:",
        )
        checkpointer = UnifiedCheckpointer(config)
        yield checkpointer
        checkpointer.close()  # Ensure proper cleanup

    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding vector."""
        return [0.1] * 768  # 768-dimensional vector

    def test_put_with_embeddings(self, checkpointer, mock_embedding) -> None:
        """Test that put() generates and stores embeddings."""
        # Mock the embedding generator
        with patch.object(
            checkpointer._embedding_generator,
            "generate_checkpoint_embedding",
            return_value=mock_embedding,
        ):
            # Create test data
            config = {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                },
            }
            checkpoint = {
                "v": 1,
                "ts": "2023-01-01T00:00:00+00:00",
                "channel_values": {"messages": ["Hello, world!"]},
                "channel_versions": {},
                "versions_seen": {},
            }
            metadata = {"step": 1, "score": 0.9}

            # Call put
            result = checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                new_versions={},
            )

            # Verify checkpoint_id was added to config
            assert "checkpoint_id" in result["configurable"]

            # Verify embedding was generated
            checkpointer._embedding_generator.generate_checkpoint_embedding.assert_called_once()

    def test_search_checkpoints(self, checkpointer) -> None:
        """Test semantic search functionality."""
        # First, store some checkpoints with embeddings
        for i in range(3):
            config = {
                "configurable": {
                    "thread_id": f"thread-{i}",
                    "checkpoint_ns": "test",
                },
            }
            checkpoint = {
                "v": 1,
                "ts": f"2023-01-0{i+1}T00:00:00+00:00",
                "channel_values": {"messages": [f"Message {i}"]},
                "channel_versions": {},
                "versions_seen": {},
            }
            metadata = {"step": i, "topic": f"topic-{i}"}

            checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                new_versions={},
            )

        # Mock the search functionality
        with patch.object(
            checkpointer._embedding_generator,
            "embeddings",
            create=True,
        ) as mock_embeddings:
            mock_embeddings.embed_query.return_value = [0.1] * 768

            with patch.object(
                checkpointer._client,
                "semantic_search",
                return_value=[
                    {
                        "thread_id": "thread-0",
                        "checkpoint_id": "test-id",
                        "checkpoint": '{"v": 1}',
                        "metadata": '{"step": 0}',
                        "_score": 0.95,
                    },
                ],
            ):
                # Search
                results = checkpointer.search_checkpoints(
                    query="find relevant messages",
                    limit=5,
                )

                # Verify results
                assert len(results) == 1
                assert results[0].config["configurable"]["thread_id"] == "thread-0"

    @pytest.mark.asyncio
    async def test_async_search_checkpoints(self, checkpointer) -> None:
        """Test async semantic search."""
        # Mock the embedding and search
        with patch.object(
            checkpointer._embedding_generator,
            "embeddings",
            create=True,
        ) as mock_embeddings:
            mock_embeddings.embed_query.return_value = [0.1] * 768

            with patch.object(
                checkpointer._client,
                "semantic_search",
                return_value=[],
            ):
                # Search asynchronously
                results = await checkpointer.asearch_checkpoints(
                    query="test query",
                    thread_id="test-thread",
                    limit=10,
                )

                # Verify empty results
                assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
