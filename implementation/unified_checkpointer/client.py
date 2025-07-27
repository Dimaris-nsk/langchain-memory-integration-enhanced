"""Unified-memory client wrapper for checkpointer."""

import base64
import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .exceptions import (
    ConnectionError as UnifiedConnectionError,
)
from .exceptions import (
    UnifiedMemoryUnavailable,
)

# Fixed namespace for checkpoint ID to UUID conversion
# This ensures deterministic UUID generation from string IDs
CHECKPOINT_NAMESPACE_UUID = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")


class UnifiedMemoryClient:
    """Wrapper around qdrant-client for unified-memory operations.

    This client provides a high-level interface for interacting with
    unified-memory's Qdrant backend, handling connection management,
    retries, and providing checkpointer-specific methods.

    Uses lazy loading pattern to avoid blocking operations during import.
    """

    def __init__(
        self,
        collection_name: str,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int = 768,
    ) -> None:
        """Initialize UnifiedMemoryClient with lazy loading.

        Args:
            collection_name: Name of the Qdrant collection to use
            url: Qdrant server URL (defaults to in-memory if None)
            api_key: API key for authentication
            vector_size: Dimension of vector embeddings
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._url = url
        self._api_key = api_key

        # Lazy loading - don't create client until first use
        self._client: QdrantClient | None = None
        self._collection_ensured = False

    @property
    def client(self) -> QdrantClient:
        """Get or create the Qdrant client lazily.

        Returns:
            QdrantClient instance

        Raises:
            ConnectionError: If connection fails
        """
        if self._client is None:
            try:
                # Initialize Qdrant client
                if self._url == ":memory:" or self._url is None:
                    # Use in-memory mode for testing
                    self._client = QdrantClient(":memory:")
                else:
                    self._client = QdrantClient(url=self._url, api_key=self._api_key)
            except Exception as e:
                msg = f"Failed to connect to unified-memory: {e}"
                raise UnifiedConnectionError(msg) from e

        # Ensure collection exists on first access
        if not self._collection_ensured:
            self._ensure_collection()
            self._collection_ensured = True

        return self._client

    def _to_jsonable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(item) for item in obj]
        if hasattr(obj, "__dict__"):
            # Convert objects to dictionaries
            return self._to_jsonable(obj.__dict__)
        # Fallback to string representation
        return str(obj)

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper configuration."""
        collections = self.client.get_collections()

        if self.collection_name not in [c.name for c in collections.collections]:
            # Create collection with vector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        else:
            pass

    def store_checkpoint(
        self, checkpoint_data: dict[str, Any], embedding: list[float] | None = None,
    ) -> str:
        """Store a checkpoint in unified-memory.

        Args:
            checkpoint_data: Checkpoint data to store, should contain:
                - thread_id: Thread identifier
                - checkpoint: The actual checkpoint data
                - metadata: Additional metadata
                - checkpoint_ns: Namespace
                - tags: Optional list of tags
                - parent_config: Optional parent configuration
            embedding: Optional pre-computed embedding vector

        Returns:
            Checkpoint ID

        Raises:
            UnifiedMemoryUnavailable: If storage fails
        """
        try:
            # Use checkpoint ID from data if provided, otherwise generate new one
            checkpoint_id = checkpoint_data.get("checkpoint_id") or str(uuid.uuid4())

            # Convert string ID to UUID for Qdrant compatibility
            qdrant_id = str(uuid.uuid5(CHECKPOINT_NAMESPACE_UUID, checkpoint_id))

            # Handle checkpoint serialization
            checkpoint = checkpoint_data.get("checkpoint", {})
            if isinstance(checkpoint, bytes):
                checkpoint = base64.b64encode(checkpoint).decode("utf-8")

            # Prepare document according to CheckpointDocument schema
            document = {
                "id": checkpoint_id,
                "thread_id": checkpoint_data.get("thread_id", ""),
                "checkpoint_ns": checkpoint_data.get("checkpoint_ns", ""),
                "checkpoint": checkpoint,
                "metadata": self._to_jsonable(checkpoint_data.get("metadata", {})),
                "parent_config": self._to_jsonable(
                    checkpoint_data.get("parent_config"),
                ),
                "tags": checkpoint_data.get("tags", []),
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "message_count": 0,  # NOTE: Extract from checkpoint
                "step_count": 0,  # NOTE: Extract from checkpoint
                "summary": None,  # NOTE: Generate summary
                "keywords": [],  # NOTE: Extract keywords
                "version": 1,
                "channel_versions": self._to_jsonable(
                    checkpoint_data.get("channel_versions", {}),
                ),
                "pending_writes": self._to_jsonable(
                    checkpoint_data.get("pending_writes", []),
                ),
            }
            # Use provided embedding or generate placeholder
            vector = embedding if embedding is not None else [0.0] * self.vector_size

            # Validate vector size
            if len(vector) != self.vector_size:
                error_msg = (
                    f"Embedding size mismatch: expected {self.vector_size}, "
                    f"got {len(vector)}"
                )
                raise ValueError(error_msg)

            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=qdrant_id,  # Use UUID instead of original string ID
                        payload=document,
                        vector=vector,
                    ),
                ],
            )

            return checkpoint_id

        except Exception as e:
            error_msg = f"Failed to store checkpoint: {e}"
            raise UnifiedMemoryUnavailable(error_msg) from e

    async def store_checkpoint_batch(
        self,
        checkpoints_data: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
        progress_callback: callable | None = None,
    ) -> list[str]:
        """Store multiple checkpoints in a single batch operation.

        Args:
            checkpoints_data: List of checkpoint data dictionaries
            embeddings: Optional list of pre-computed embedding vectors
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of checkpoint IDs

        Raises:
            UnifiedMemoryUnavailable: If storage fails
        """
        try:
            checkpoint_ids = []
            points = []
            # Process each checkpoint
            for idx, checkpoint_data in enumerate(checkpoints_data):
                # Generate unique checkpoint ID
                checkpoint_id = str(uuid.uuid4())
                checkpoint_ids.append(checkpoint_id)

                # Handle checkpoint serialization
                checkpoint = checkpoint_data.get("checkpoint", {})
                if isinstance(checkpoint, bytes):
                    checkpoint = base64.b64encode(checkpoint).decode("utf-8")

                # Prepare document
                document = {
                    "id": checkpoint_id,
                    "thread_id": checkpoint_data.get("thread_id", ""),
                    "checkpoint_ns": checkpoint_data.get("checkpoint_ns", ""),
                    "checkpoint": checkpoint,
                    "metadata": self._to_jsonable(checkpoint_data.get("metadata", {})),
                    "parent_config": self._to_jsonable(
                        checkpoint_data.get("parent_config"),
                    ),
                    "tags": checkpoint_data.get("tags", []),
                    "created_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "message_count": 0,
                    "step_count": 0,
                    "summary": None,
                    "keywords": [],
                    "version": 1,
                    "channel_versions": self._to_jsonable(
                        checkpoint_data.get("channel_versions", {}),
                    ),
                    "pending_writes": self._to_jsonable(
                        checkpoint_data.get("pending_writes", []),
                    ),
                }

                # Use provided embedding or generate placeholder
                if embeddings and idx < len(embeddings):
                    vector = embeddings[idx]
                else:
                    vector = [0.0] * self.vector_size

                # Validate vector size
                if len(vector) != self.vector_size:
                    msg = (
                        f"Embedding size mismatch at index {idx}: "
                        f"expected {self.vector_size}, got {len(vector)}"
                    )
                    raise ValueError(msg)

                # Create point
                points.append(
                    PointStruct(
                        id=checkpoint_id,
                        payload=document,
                        vector=vector,
                    ),
                )
            qdrant_ids = [
                str(uuid.uuid5(CHECKPOINT_NAMESPACE_UUID, cid))
                for cid in checkpoint_ids
            ]

            # Batch retrieve from Qdrant
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=qdrant_ids,
            )

            # Map results by qdrant_id
            result_map = {point.id: point.payload for point in results}
            checkpoints = []

            # Map back to original checkpoint_ids
            for _idx, (_checkpoint_id, qdrant_id) in enumerate(
                zip(checkpoint_ids, qdrant_ids, strict=False),
            ):
                if qdrant_id in result_map:
                    payload = result_map[qdrant_id]

                    # Decode checkpoint if base64 encoded
                    checkpoint = payload.get("checkpoint", "")
                    if checkpoint and isinstance(checkpoint, str):
                        try:
                            checkpoint = base64.b64decode(checkpoint)
                        except Exception:
                            pass  # Not base64 encoded

                    payload["checkpoint"] = checkpoint
                    checkpoints.append(payload)
                else:
                    checkpoints.append(None)

            return checkpoints

        except Exception as e:
            msg = f"Failed to get checkpoint batch: {e}"
            raise UnifiedMemoryUnavailable(
                msg,
            ) from e

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Retrieve a checkpoint from unified-memory.

        Args:
            checkpoint_id: ID of the checkpoint

        Returns:
            Checkpoint data if found, None otherwise

        Raises:
            UnifiedMemoryUnavailable: If retrieval fails
        """
        try:
            # Convert string ID to UUID for Qdrant compatibility
            qdrant_id = str(uuid.uuid5(CHECKPOINT_NAMESPACE_UUID, checkpoint_id))

            # Retrieve from Qdrant
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[qdrant_id],  # Use UUID instead of original string ID
            )

            if not result:
                return None

            # Get the payload
            payload = result[0].payload

            # Decode checkpoint if it's base64 encoded
            if isinstance(payload.get("checkpoint"), str):
                try:
                    payload["checkpoint"] = base64.b64decode(payload["checkpoint"])
                except Exception:
                    # If decoding fails, keep as is
                    pass

            # Return the payload which contains all checkpoint data
            return payload

        except Exception as e:
            error_msg = f"Failed to retrieve checkpoint: {e}"
            raise UnifiedMemoryUnavailable(error_msg) from e