"""Embedding generation for UnifiedCheckpointer.

This module provides functionality to generate vector embeddings from checkpoint data
for semantic search capabilities.
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_openai import OpenAIEmbeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "text-embedding-3-small"  # Cheaper, good for most use cases
    dimensions: int | None = 1536  # Default OpenAI dimensions
    api_key: str | None = None
    batch_size: int = 100  # For batch processing
    cache_embeddings: bool = True
    fallback_to_summary: bool = True  # If full content too large


class EmbeddingGenerator:
    """Generates embeddings for checkpoint data."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the embedding generator.

        Args:
            config: Configuration for embeddings. If None, uses defaults.
        """
        self.config = config or EmbeddingConfig()

        # Initialize embeddings model
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OpenAI API key found. Embeddings will be disabled.")
            self.embeddings: Embeddings | None = None
        else:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.model,
                openai_api_key=api_key,
                dimensions=self.config.dimensions,
            )

        # Cache for embeddings
        self._cache = {} if self.config.cache_embeddings else None

    def generate_checkpoint_embedding(
        self, checkpoint: dict[str, Any], metadata: dict[str, Any] | None = None,
    ) -> list[float] | None:
        """Generate embedding for a checkpoint.

        Args:
            checkpoint: The checkpoint data
            metadata: Optional metadata to include

        Returns:
            Embedding vector or None if embeddings disabled
        """
        if not self.embeddings:
            return None

        # Extract text content for embedding
        text = self._extract_text_for_embedding(checkpoint, metadata)

        # Check cache
        if self._cache is not None:
            cache_key = hash(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(text)

            # Cache result
            if self._cache is not None:
                self._cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.exception(f"Failed to generate embedding: {e}")
            return None

    def _extract_text_for_embedding(
        self, checkpoint: dict[str, Any], metadata: dict[str, Any] | None = None,
    ) -> str:
        """Extract relevant text from checkpoint for embedding.

        Args:
            checkpoint: The checkpoint data
            metadata: Optional metadata

        Returns:
            Text string for embedding
        """
        parts = []

        # Extract from channel_values
        channel_values = checkpoint.get("channel_values", {})

        # Handle messages if present
        if "messages" in channel_values:
            messages = channel_values["messages"]
            if isinstance(messages, list):
                # Extract last N messages for context
                recent_messages = messages[-10:]  # Last 10 messages
                for msg in recent_messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if content:
                            parts.append(content)
                    elif isinstance(msg, str):
                        parts.append(msg)

        # Add metadata context
        if metadata:
            # Add relevant metadata fields
            for key in ["task", "description", "tags", "summary"]:
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, list):
                        parts.append(" ".join(str(v) for v in value))
                    else:
                        parts.append(str(value))

        # If content too large, summarize
        text = "\n".join(parts)
        if len(text) > 8000 and self.config.fallback_to_summary:
            # TODO: Implement summarization
            text = text[:8000] + "..."

        return text

    def search_similar_checkpoints(
        self, query: str, embeddings_list: list[list[float]], top_k: int = 5,
    ) -> list[int]:
        """Find most similar checkpoints to a query.

        Args:
            query: Search query
            embeddings_list: List of checkpoint embeddings
            top_k: Number of results to return

        Returns:
            Indices of most similar checkpoints
        """
        if not self.embeddings or not embeddings_list:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Calculate similarities (cosine similarity)
            import numpy as np

            query_vec = np.array(query_embedding)
            similarities = []

            for idx, checkpoint_embedding in enumerate(embeddings_list):
                checkpoint_vec = np.array(checkpoint_embedding)

                # Cosine similarity
                similarity = np.dot(query_vec, checkpoint_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(checkpoint_vec)
                )
                similarities.append((idx, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top k indices
            return [idx for idx, _ in similarities[:top_k]]

        except Exception as e:
            logger.exception(f"Failed to search similar checkpoints: {e}")
            return []

    def batch_generate_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (None for failed ones)
        """
        if not self.embeddings:
            return [None] * len(texts)

        results = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                results.extend(batch_embeddings)
            except Exception as e:
                logger.exception(f"Failed to generate batch embeddings: {e}")
                results.extend([None] * len(batch))

        return results