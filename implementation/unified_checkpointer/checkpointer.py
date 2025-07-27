"""UnifiedCheckpointer - LangGraph checkpointer with unified-memory backend."""

import asyncio
import base64
import builtins
import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointMetadata,
    CheckpointTuple,
)

from .cache import CheckpointCache
from .client import UnifiedMemoryClient
from .config import UnifiedCheckpointerConfig
from .connection_pool import ConnectionPool
from .embeddings import EmbeddingConfig, EmbeddingGenerator
from .exceptions import (
    ConfigurationError,
    UnifiedCheckpointerError,
)

logger = logging.getLogger(__name__)


class UnifiedCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpointer with unified-memory backend.

    This checkpointer provides full compatibility with LangGraph's
    BaseCheckpointSaver interface while adding enhanced features
    like semantic search, tagging, and analytics through the
    unified-memory backend.
    """

    def __init__(
        self,
        config: UnifiedCheckpointerConfig | None = None,
        client: UnifiedMemoryClient | None = None,
    ) -> None:
        """Initialize UnifiedCheckpointer.

        Args:
            config: Configuration object. If None, uses defaults.
            client: Optional pre-configured UnifiedMemoryClient. If provided,
                    it will be used instead of creating a new one. Useful for
                    testing with in-memory clients.
        """
        super().__init__()
        self.config = config or UnifiedCheckpointerConfig()

        # Initialize connection pool if enabled
        if self.config.pool_enabled:
            # If client is provided, pass it to ConnectionPool
            self._pool = ConnectionPool(
                collection_name=self.config.collection_name,
                url=self.config.unified_memory_url,
                api_key=self.config.api_key,
                vector_size=768,  # Match embedding dimensions
                min_connections=self.config.pool_min_connections,
                max_connections=self.config.pool_max_connections,
                connection_ttl=self.config.pool_connection_ttl,
                idle_timeout=self.config.pool_idle_timeout,
                health_check_interval=self.config.pool_health_check_interval,
                is_async=False,  # Use sync clients for now
                client=client,  # Pass the client if provided
            )
            # Don't initialize pool here - it will be initialized on first use
            self._pool_initialized = False
            self._client = client  # Store the provided client if any
        else:
            # Fallback to direct client
            self._pool = None
            # Use provided client or create new one
            if client is not None:
                self._client = client
            else:
                self._client = UnifiedMemoryClient(
                    collection_name=self.config.collection_name,
                    url=self.config.unified_memory_url,
                    api_key=self.config.api_key,
                )

        # Initialize embedding generator
        embedding_config = EmbeddingConfig(
            model="text-embedding-3-small",
            dimensions=768,  # Match client vector_size
            api_key=self.config.api_key,
        )
        self._embedding_generator = EmbeddingGenerator(embedding_config)

        # Initialize serializer
        self.serde = self.config.serializer

        # Initialize cache
        if self.config.cache_enabled:
            self._cache = CheckpointCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
        else:
            self._cache = None

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not self.config.collection_name:
            msg = "collection_name cannot be empty"
            raise ConfigurationError(msg)

        if self.config.cache_size < 0:
            msg = "cache_size must be non-negative"
            raise ConfigurationError(msg)

        if self.config.connection_pool_size < 1:
            msg = "connection_pool_size must be at least 1"
            raise ConfigurationError(msg)

        if self.config.batch_size < 1:
            msg = "batch_size must be at least 1"
            raise ConfigurationError(msg)

        if self.config.retry_attempts < 1:
            msg = "retry_attempts must be at least 1"
            raise ConfigurationError(msg)

        if self.config.compression_threshold_kb < 0:
            msg = "compression_threshold_kb must be non-negative"
            raise ConfigurationError(msg)

    def _ensure_pool_initialized(self) -> None:
        """Ensure the connection pool is initialized.

        This method performs lazy initialization of the pool
        to avoid event loop issues when the checkpointer is
        created outside of an async context.
        """
        if self._pool and not self._pool_initialized:
            try:
                # Initialize the pool synchronously
                asyncio.run(self._pool.initialize())
                self._pool_initialized = True
                logger.info("Connection pool initialized successfully")
            except Exception as e:
                logger.exception(f"Failed to initialize connection pool: {e}")
                # Fall back to direct client if pool initialization fails
                self._pool = None
                self._client = UnifiedMemoryClient(
                    collection_name=self.config.collection_name,
                    url=self.config.unified_memory_url,
                    api_key=self.config.api_key,
                    vector_size=768,
                )
                logger.info("Falling back to direct client connection")

    @contextmanager
    def _get_client(self):
        """Get a client from pool or direct client.

        This method abstracts the difference between using a connection pool
        and a direct client connection.
        """
        # For sync operations, always use direct client to avoid event loop issues
        # Pool is only used for async operations
        if self._client is None:
            # Initialize client if it's None (can happen in tests or edge cases)
            logger.warning("Client was None in _get_client, initializing new client")
            self._client = UnifiedMemoryClient(
                collection_name=self.config.collection_name,
                url=self.config.unified_memory_url,
                api_key=self.config.api_key,
                vector_size=768,
            )
        yield self._client

    # Required sync methods from BaseCheckpointSaver

    def put(
        self,
        config: RunnableConfig,
        checkpoint: dict[str, Any],
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration, metadata, and new versions.

        Args:
            config: The runnable configuration containing thread_id
            checkpoint: The checkpoint data to store
            metadata: Additional metadata for the checkpoint
            new_versions: Channel versions for the checkpoint

        Returns:
            Updated configuration with checkpoint_id
        """
        # Extract thread_id from config
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            msg = "thread_id is required in config"
            raise ConfigurationError(msg)

        # Extract checkpoint_ns (namespace)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Convert Checkpoint object to dict if needed
        if hasattr(checkpoint, "__dict__"):
            # It's an object (like Checkpoint from langgraph), convert to dict
            checkpoint_dict = {
                "v": getattr(checkpoint, "v", 1),
                "id": getattr(checkpoint, "id", str(uuid.uuid4())),
                "ts": getattr(checkpoint, "ts", ""),
                "channel_values": getattr(checkpoint, "channel_values", {}),
                "channel_versions": getattr(checkpoint, "channel_versions", {}),
                "versions_seen": getattr(checkpoint, "versions_seen", {}),
            }
            checkpoint_id = checkpoint_dict["id"]
        else:
            # It's already a dict
            checkpoint_dict = checkpoint
            checkpoint_id = checkpoint_dict.get("id") or str(uuid.uuid4())

            # Update checkpoint with id if needed
            if "id" not in checkpoint_dict:
                checkpoint_dict = {**checkpoint_dict, "id": checkpoint_id}

        # Serialize checkpoint data
        serialized_checkpoint = self.serde.dumps(checkpoint_dict)
        serialized_metadata = self.serde.dumps(metadata)

        # Generate embedding for semantic search
        embedding = self._embedding_generator.generate_checkpoint_embedding(
            checkpoint=checkpoint_dict, metadata=metadata,
        )

        # Prepare checkpoint document
        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint": serialized_checkpoint,
            "metadata": serialized_metadata,
            "parent_config": config.get("configurable", {}).get("checkpoint_id"),
            "channel_versions": dict(new_versions) if new_versions else {},
            "tags": [],  # TODO: Extract from metadata
        }

        # Store checkpoint using client from pool or direct
        with self._get_client() as client:
            # Store checkpoint (synchronously for now)
            # TODO: Make this properly async
            stored_id = client.store_checkpoint(checkpoint_data, embedding=embedding)
            logger.debug(
                f"Stored checkpoint with ID: {stored_id}, expected: {checkpoint_id}",
            )

        # Invalidate cache for this thread
        if self._cache:
            self._cache.invalidate(thread_id)
            logger.debug(f"Cache invalidated for thread_id={thread_id}")

        # Return updated config with checkpoint_id
        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint_id,
            },
        }

    def put_batch(
        self,
        configs: list[RunnableConfig],
        checkpoints: list[dict[str, Any]],
        metadatas: list[CheckpointMetadata],
        new_versions_list: list[ChannelVersions],
        progress_callback: callable | None = None,
    ) -> list[RunnableConfig]:
        """Store multiple checkpoints in a single batch operation.

        Args:
            configs: List of runnable configurations
            checkpoints: List of checkpoint data
            metadatas: List of metadata for checkpoints
            new_versions_list: List of channel versions
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of updated configurations with checkpoint_ids

        Raises:
            ValueError: If input lists have different lengths
        """
        # Validate input lengths
        if not (
            len(configs) == len(checkpoints) == len(metadatas) == len(new_versions_list)
        ):
            msg = "All input lists must have the same length"
            raise ValueError(msg)

        checkpoint_data_list = []
        embeddings = []
        updated_configs = []

        # Process each checkpoint
        for idx, (config, checkpoint, metadata, new_versions) in enumerate(
            zip(configs, checkpoints, metadatas, new_versions_list, strict=False),
        ):
            # Extract thread_id from config
            thread_id = config["configurable"].get("thread_id")
            if not thread_id:
                msg = f"thread_id is required in config at index {idx}"
                raise ConfigurationError(msg)

            # Extract checkpoint_ns
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

            # Generate checkpoint_id if not present
            checkpoint_id = checkpoint.get("id") or str(uuid.uuid4())

            # Update checkpoint with id if needed
            if "id" not in checkpoint:
                checkpoint = {**checkpoint, "id": checkpoint_id}

            # Serialize checkpoint data
            serialized_checkpoint = self.serde.dumps(checkpoint)
            serialized_metadata = self.serde.dumps(metadata)

            # Generate embedding
            embedding = self._embedding_generator.generate_checkpoint_embedding(
                checkpoint=checkpoint, metadata=metadata,
            )
            embeddings.append(embedding)

            # Prepare checkpoint document
            checkpoint_data = {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint": serialized_checkpoint,
                "metadata": serialized_metadata,
                "parent_config": config.get("configurable", {}).get("checkpoint_id"),
                "channel_versions": dict(new_versions) if new_versions else {},
                "tags": [],
            }
            checkpoint_data_list.append(checkpoint_data)

            # Prepare updated config
            updated_config = {
                **config,
                "configurable": {
                    **config.get("configurable", {}),
                    "checkpoint_id": checkpoint_id,
                },
            }
            updated_configs.append(updated_config)

        # Store batch using connection pool
        with self._get_client() as client:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    client.store_checkpoint_batch(
                        checkpoint_data_list,
                        embeddings=embeddings,
                        progress_callback=progress_callback,
                    ),
                )
            finally:
                loop.close()

        # Invalidate cache for all threads
        if self._cache:
            thread_ids = {config["configurable"]["thread_id"] for config in configs}
            for thread_id in thread_ids:
                self._cache.invalidate(thread_id)
            logger.debug(f"Cache invalidated for {len(thread_ids)} threads")

        return updated_configs

    def get_batch(self, configs: list[RunnableConfig]) -> list[CheckpointTuple | None]:
        """Retrieve multiple checkpoints in a single batch operation.

        Args:
            configs: List of runnable configurations

        Returns:
            List of CheckpointTuples (None for missing checkpoints)

        Raises:
            ConfigurationError: If thread_id is missing in any config
        """
        # Extract checkpoint IDs
        checkpoint_ids = []
        config_map = {}

        for idx, config in enumerate(configs):
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id")

            if not thread_id:
                msg = f"thread_id is required in config at index {idx}"
                raise ConfigurationError(msg)

            checkpoint_id = configurable.get("checkpoint_id")
            if checkpoint_id:
                checkpoint_ids.append(checkpoint_id)
                config_map[checkpoint_id] = (config, idx)
            else:
                # If no checkpoint_id, we'll need to fetch latest
                # For now, just skip (TODO: implement batch latest fetch)
                checkpoint_ids.append(None)

        # Batch retrieve checkpoints using connection pool
        with self._get_client() as client:
            loop = asyncio.new_event_loop()
            try:
                checkpoints_data = loop.run_until_complete(
                    client.get_checkpoint_batch(
                        [cid for cid in checkpoint_ids if cid is not None],
                    ),
                )
            finally:
                loop.close()

        # Process results
        result_map = {}
        for checkpoint_data in checkpoints_data:
            if checkpoint_data:
                checkpoint_id = checkpoint_data["id"]
                result_map[checkpoint_id] = checkpoint_data

        # Build result list
        results = []
        for idx, config in enumerate(configs):
            checkpoint_id = checkpoint_ids[idx]

            if checkpoint_id and checkpoint_id in result_map:
                checkpoint_data = result_map[checkpoint_id]

                # Deserialize checkpoint
                checkpoint = self.serde.loads(checkpoint_data["checkpoint"])
                metadata = self.serde.loads(checkpoint_data["metadata"])

                # Create parent config if exists
                parent_config = None
                if checkpoint_data.get("parent_config"):
                    parent_config = {
                        **config,
                        "configurable": {
                            **config.get("configurable", {}),
                            "checkpoint_id": checkpoint_data["parent_config"],
                        },
                    }

                # Create CheckpointTuple
                tuple_result = CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                    pending_writes=checkpoint_data.get("pending_writes", []),
                )
                results.append(tuple_result)
            else:
                results.append(None)

        return results

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint tuple for the given configuration.

        Args:
            config: The runnable configuration containing thread_id and optionally checkpoint_id

        Returns:
            CheckpointTuple if found, None otherwise
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id:
            msg = "thread_id is required in config"
            raise ConfigurationError(msg)

        # Check cache first if enabled
        if self._cache:
            cached_tuple = self._cache.get(thread_id, checkpoint_id)
            if cached_tuple:
                logger.debug(
                    f"Cache hit for thread_id={thread_id}, checkpoint_id={checkpoint_id}",
                )
                return cached_tuple

        # If no checkpoint_id specified, get the latest checkpoint
        if not checkpoint_id:
            # Get list of checkpoints for this thread
            with self._get_client() as client:
                checkpoints = client.list_checkpoints(thread_id=thread_id, limit=1)

            if not checkpoints:
                return None

            # Use the most recent checkpoint
            checkpoint_data = checkpoints[0]
        else:
            # Get specific checkpoint by ID
            with self._get_client() as client:
                checkpoint_data = client.get_checkpoint(checkpoint_id)

            if not checkpoint_data:
                return None

            # Verify it belongs to the requested thread
            if checkpoint_data.get("thread_id") != thread_id:
                return None

        # Deserialize checkpoint data
        checkpoint_data_raw = checkpoint_data.get("checkpoint", {})

        # If checkpoint is a base64 string, decode it first
        if isinstance(checkpoint_data_raw, str):
            try:
                checkpoint_bytes = base64.b64decode(checkpoint_data_raw)
            except Exception as e:
                msg = f"Failed to decode checkpoint base64: {e}"
                raise UnifiedCheckpointerError(msg)
        elif isinstance(checkpoint_data_raw, bytes):
            checkpoint_bytes = checkpoint_data_raw
        else:
            # Fallback for dict (shouldn't happen in production)
            checkpoint_dict = checkpoint_data_raw
            checkpoint_bytes = None

        # If we have bytes, deserialize them
        if checkpoint_bytes is not None:
            try:
                # Deserialize checkpoint bytes
                checkpoint_dict = self.serde.loads(checkpoint_bytes)
                # Debug: Check what we got
                logger.debug(
                    f"Deserialized checkpoint type: {type(checkpoint_dict)}, keys: {list(checkpoint_dict.keys()) if isinstance(checkpoint_dict, dict) else 'not a dict'}",
                )
            except Exception as e:
                msg = f"Failed to deserialize checkpoint: {e}"
                raise UnifiedCheckpointerError(msg)

        # Deserialize channel_values if they were serialized
        if "channel_values" in checkpoint_dict and isinstance(
            checkpoint_dict["channel_values"], (str, bytes),
        ):
            try:
                # If it's a string with type:data format (legacy)
                if (
                    isinstance(checkpoint_dict["channel_values"], str)
                    and ":" in checkpoint_dict["channel_values"]
                ):
                    serde_type, serde_data = checkpoint_dict["channel_values"].split(
                        ":", 1,
                    )
                    channel_values = self.serde.loads_typed((serde_type, serde_data))
                else:
                    # Otherwise use regular loads
                    channel_values = self.serde.loads(checkpoint_dict["channel_values"])
                checkpoint_dict["channel_values"] = channel_values
            except Exception as e:
                msg = f"Failed to deserialize channel_values: {e}"
                raise UnifiedCheckpointerError(msg)

        # Debug: Check checkpoint_dict before processing
        logger.debug(
            f"checkpoint_dict type: {type(checkpoint_dict)}, keys: {list(checkpoint_dict.keys()) if isinstance(checkpoint_dict, dict) else 'not a dict'}",
        )

        # Checkpoint in LangGraph is always a dict, not an object
        checkpoint = checkpoint_dict
        if not isinstance(checkpoint, dict):
            msg = f"Expected checkpoint to be a dict, got {type(checkpoint)}"
            raise UnifiedCheckpointerError(msg)

        # Ensure checkpoint has required fields
        if "id" not in checkpoint:
            msg = "Checkpoint missing required field 'id'"
            raise UnifiedCheckpointerError(msg)

        logger.debug(f"Processing checkpoint with id: {checkpoint.get('id')}")

        # Create metadata
        metadata = CheckpointMetadata(
            source="unified-memory",
            write_time=checkpoint_data.get("created_at"),
            step=checkpoint_dict.get("v", 1),
        )

        # Create parent config if parent_id exists
        parent_config = None
        parent_id = checkpoint_data.get("parent_checkpoint_id")
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                },
            }

        # Get pending writes for this checkpoint
        pending_writes = []
        try:
            raw_writes = self._client.get_pending_writes(checkpoint["id"])

            # Deserialize pending writes
            for task_id, channel, raw_value in raw_writes:
                try:
                    # Deserialize the value
                    if isinstance(raw_value, str) and raw_value:
                        try:
                            # Try to decode as base64
                            data_bytes = base64.b64decode(raw_value)
                            value = self.serde.loads(data_bytes)
                        except:
                            # If not base64, use as is
                            value = raw_value
                    elif (
                        isinstance(raw_value, dict)
                        and "type" in raw_value
                        and "data" in raw_value
                    ):
                        # Legacy format with type info
                        type_str = raw_value["type"]
                        data_bytes = base64.b64decode(raw_value["data"])
                        value = self.serde.loads_typed((type_str, data_bytes))
                    else:
                        value = raw_value
                    pending_writes.append((task_id, channel, value))
                except Exception as e:
                    # Log error but continue processing other writes
                    import logging

                    logging.warning(
                        f"Failed to deserialize pending write {task_id}: {e}",
                    )
                    pending_writes.append((task_id, channel, raw_value))
        except Exception as e:
            # If we can't get pending writes, just continue without them
            import logging

            logging.warning(
                f"Failed to get pending writes for checkpoint {checkpoint['id']}: {e}",
            )

        # Return CheckpointTuple
        checkpoint_tuple = CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                },
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )

        # Store in cache if enabled
        if self._cache:
            self._cache.put(thread_id, checkpoint_tuple, checkpoint["id"])

        return checkpoint_tuple

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the given criteria.

        Args:
            config: The runnable configuration for filtering by thread
            filter: Additional filter criteria
            before: List checkpoints before this configuration
            limit: Maximum number of results to return

        Returns:
            Iterator of matching checkpoint tuples
        """
        # TODO: Implement checkpoint listing
        # Extract thread_id from config if provided
        thread_id = None
        if config:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id")

        # Get checkpoints from client
        with self._get_client() as client:
            checkpoints = client.list_checkpoints(
                thread_id=thread_id, limit=limit or 100,
            )

        # Convert each checkpoint to CheckpointTuple
        for checkpoint_data in checkpoints:
            # Create config for get_tuple
            checkpoint_config = {
                "configurable": {
                    "thread_id": checkpoint_data.get("thread_id"),
                    "checkpoint_id": checkpoint_data.get("id"),
                },
            }

            # Use get_tuple to convert to CheckpointTuple
            tuple_result = self.get_tuple(checkpoint_config)
            if tuple_result:
                yield tuple_result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint.

        Args:
            config: The runnable configuration
            writes: Sequence of (channel, value) tuples to write
            task_id: Identifier for the task performing the writes
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id or not checkpoint_id:
            msg = "thread_id and checkpoint_id are required for put_writes"
            raise ConfigurationError(msg)

        # Store each pending write
        with self._get_client() as client:
            for channel, value in writes:
                # Serialize the value
                if value is not None:
                    data_bytes = self.serde.dumps(value)
                    # Encode bytes to base64 for JSON storage
                    serialized_value = base64.b64encode(data_bytes).decode("utf-8")
                else:
                    serialized_value = None

                # Store in Qdrant
                client.store_pending_write(
                    checkpoint_id=checkpoint_id,
                    thread_id=thread_id,
                    task_id=task_id,
                    channel=channel,
                    value=serialized_value,
                )

    # Required async methods from BaseCheckpointSaver

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: dict[str, Any],
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put method.

        Stores a checkpoint with metadata asynchronously.

        Args:
            config: The runnable configuration
            checkpoint: The checkpoint to store
            metadata: Additional metadata
            new_versions: New channel versions

        Returns:
            Updated configuration with new checkpoint_id
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        if not thread_id:
            msg = "thread_id is required in config"
            raise ConfigurationError(msg)

        # Convert Checkpoint object to dict if needed
        if hasattr(checkpoint, "__dict__"):
            # It's an object (like Checkpoint from langgraph), convert to dict
            checkpoint_dict = {
                "v": getattr(checkpoint, "v", 1),
                "id": getattr(checkpoint, "id", str(uuid.uuid4())),
                "ts": getattr(checkpoint, "ts", ""),
                "channel_values": getattr(checkpoint, "channel_values", {}),
                "channel_versions": getattr(checkpoint, "channel_versions", {}),
                "versions_seen": getattr(checkpoint, "versions_seen", {}),
            }
            checkpoint_id = checkpoint_dict["id"]
        else:
            # It's already a dict
            checkpoint_dict = checkpoint
            checkpoint_id = checkpoint_dict.get("id") or str(uuid.uuid4())

            # Update checkpoint with id if needed
            if "id" not in checkpoint_dict:
                checkpoint_dict = {**checkpoint_dict, "id": checkpoint_id}

        # Serialize checkpoint data
        serialized_checkpoint = self.serde.dumps(checkpoint_dict)
        serialized_metadata = self.serde.dumps(metadata)

        # Generate embedding for semantic search
        embedding = self._embedding_generator.generate_checkpoint_embedding(
            checkpoint=checkpoint_dict, metadata=metadata,
        )

        # Prepare checkpoint document
        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint": serialized_checkpoint,
            "metadata": serialized_metadata,
            "parent_config": config.get("configurable", {}).get("checkpoint_id"),
            "channel_versions": dict(new_versions) if new_versions else {},
            "tags": [],  # TODO: Extract from metadata
        }

        # Store checkpoint asynchronously using connection pool
        if self._pool:
            async with self._pool.acquire() as client:
                await client.astore_checkpoint(checkpoint_data, embedding=embedding)
        else:
            await self._client.astore_checkpoint(checkpoint_data, embedding=embedding)

        # Invalidate cache for this thread
        if self._cache:
            self._cache.invalidate(thread_id)
            logger.debug(f"Cache invalidated for thread_id={thread_id}")

        # Return updated config with checkpoint_id
        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint_id,
            },
        }

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of get_tuple method.

        Retrieve a checkpoint tuple for the given configuration asynchronously.

        Args:
            config: The runnable configuration

        Returns:
            CheckpointTuple if found, None otherwise
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id:
            msg = "thread_id is required in config"
            raise ConfigurationError(msg)

        # Check cache first if enabled
        if self._cache:
            cached_tuple = self._cache.get(thread_id, checkpoint_id)
            if cached_tuple:
                logger.debug(
                    f"Cache hit for thread_id={thread_id}, checkpoint_id={checkpoint_id}",
                )
                return cached_tuple

        # If no checkpoint_id specified, get the latest checkpoint
        if not checkpoint_id:
            # Get list of checkpoints for this thread
            if self._pool:
                async with self._pool.acquire() as client:
                    checkpoints = await client.alist_checkpoints(
                        thread_id=thread_id, limit=1,
                    )
            else:
                checkpoints = await self._client.alist_checkpoints(
                    thread_id=thread_id, limit=1,
                )

            if not checkpoints:
                return None

            # Use the most recent checkpoint
            checkpoint_data = checkpoints[0]
        else:
            # Get specific checkpoint by ID
            if self._pool:
                async with self._pool.acquire() as client:
                    checkpoint_data = await client.aget_checkpoint(checkpoint_id)
            else:
                checkpoint_data = await self._client.aget_checkpoint(checkpoint_id)

            if not checkpoint_data:
                return None

            # Verify it belongs to the requested thread
            if checkpoint_data.get("thread_id") != thread_id:
                return None

        # Deserialize checkpoint data
        checkpoint_data_raw = checkpoint_data.get("checkpoint", {})

        # If checkpoint is a base64 string, decode it first
        if isinstance(checkpoint_data_raw, str):
            try:
                checkpoint_bytes = base64.b64decode(checkpoint_data_raw)
            except Exception as e:
                msg = f"Failed to decode checkpoint base64: {e}"
                raise UnifiedCheckpointerError(msg)
        elif isinstance(checkpoint_data_raw, bytes):
            checkpoint_bytes = checkpoint_data_raw
        else:
            # Fallback for dict (shouldn't happen in production)
            checkpoint_dict = checkpoint_data_raw
            checkpoint_bytes = None

        # If we have bytes, deserialize them
        if checkpoint_bytes is not None:
            try:
                # Deserialize checkpoint bytes
                checkpoint_dict = self.serde.loads(checkpoint_bytes)
                # Debug: Check what we got
                logger.debug(
                    f"Deserialized checkpoint type: {type(checkpoint_dict)}, keys: {list(checkpoint_dict.keys()) if isinstance(checkpoint_dict, dict) else 'not a dict'}",
                )
            except Exception as e:
                msg = f"Failed to deserialize checkpoint: {e}"
                raise UnifiedCheckpointerError(msg)

        # Deserialize channel_values if they were serialized
        if "channel_values" in checkpoint_dict and isinstance(
            checkpoint_dict["channel_values"], (str, bytes),
        ):
            try:
                # If it's a string with type:data format (legacy)
                if (
                    isinstance(checkpoint_dict["channel_values"], str)
                    and ":" in checkpoint_dict["channel_values"]
                ):
                    serde_type, serde_data = checkpoint_dict["channel_values"].split(
                        ":", 1,
                    )
                    channel_values = self.serde.loads_typed((serde_type, serde_data))
                else:
                    # Otherwise use regular loads
                    channel_values = self.serde.loads(checkpoint_dict["channel_values"])
                checkpoint_dict["channel_values"] = channel_values
            except Exception as e:
                msg = f"Failed to deserialize channel_values: {e}"
                raise UnifiedCheckpointerError(msg)

        # Debug: Check checkpoint_dict before processing
        logger.debug(
            f"checkpoint_dict type: {type(checkpoint_dict)}, keys: {list(checkpoint_dict.keys()) if isinstance(checkpoint_dict, dict) else 'not a dict'}",
        )

        # Checkpoint in LangGraph is always a dict, not an object
        checkpoint = checkpoint_dict
        if not isinstance(checkpoint, dict):
            msg = f"Expected checkpoint to be a dict, got {type(checkpoint)}"
            raise UnifiedCheckpointerError(msg)

        # Ensure checkpoint has required fields
        if "id" not in checkpoint:
            msg = "Checkpoint missing required field 'id'"
            raise UnifiedCheckpointerError(msg)

        logger.debug(f"Processing checkpoint with id: {checkpoint.get('id')}")

        # Create metadata
        metadata = CheckpointMetadata(
            source="unified-memory",
            write_time=checkpoint_data.get("created_at"),
            step=checkpoint_dict.get("v", 1),
        )

        # Create parent config if parent_id exists
        parent_config = None
        parent_id = checkpoint_data.get("parent_checkpoint_id")
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                },
            }

        # Get pending writes for this checkpoint
        pending_writes = []
        try:
            if self._pool:
                async with self._pool.acquire() as client:
                    raw_writes = await client.aget_pending_writes(checkpoint["id"])
            else:
                raw_writes = await self._client.aget_pending_writes(checkpoint["id"])

            # Deserialize pending writes
            for task_id, channel, raw_value in raw_writes:
                try:
                    # Deserialize the value
                    if isinstance(raw_value, str) and raw_value:
                        try:
                            # Try to decode as base64
                            data_bytes = base64.b64decode(raw_value)
                            value = self.serde.loads(data_bytes)
                        except:
                            # If not base64, use as is
                            value = raw_value
                    elif (
                        isinstance(raw_value, dict)
                        and "type" in raw_value
                        and "data" in raw_value
                    ):
                        # Legacy format with type info
                        type_str = raw_value["type"]
                        data_bytes = base64.b64decode(raw_value["data"])
                        value = self.serde.loads_typed((type_str, data_bytes))
                    else:
                        value = raw_value
                    pending_writes.append((task_id, channel, value))
                except Exception as e:
                    # Log error but continue processing other writes
                    logging.warning(
                        f"Failed to deserialize pending write {task_id}: {e}",
                    )
                    pending_writes.append((task_id, channel, raw_value))
        except Exception as e:
            # If we can't get pending writes, just continue without them
            logging.warning(
                f"Failed to get pending writes for checkpoint {checkpoint['id']}: {e}",
            )

        # Return CheckpointTuple
        checkpoint_tuple = CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                },
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )

        # Store in cache if enabled
        if self._cache:
            self._cache.put(thread_id, checkpoint_tuple, checkpoint["id"])

        return checkpoint_tuple

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list method.

        List checkpoints matching the given criteria asynchronously.

        Args:
            config: The runnable configuration for filtering by thread
            filter: Additional filter criteria
            before: List checkpoints before this configuration
            limit: Maximum number of results to return

        Returns:
            Async iterator of matching checkpoint tuples
        """
        # Extract thread_id from config if provided
        thread_id = None
        if config:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id")

        # Get checkpoints from client asynchronously
        if self._pool:
            async with self._pool.acquire() as client:
                checkpoints = await client.alist_checkpoints(
                    thread_id=thread_id,
                    limit=limit or 100,
                    offset=0,  # TODO: Implement proper pagination with before
                )
        else:
            checkpoints = await self._client.alist_checkpoints(
                thread_id=thread_id,
                limit=limit or 100,
                offset=0,  # TODO: Implement proper pagination with before
            )

        # Convert each checkpoint to CheckpointTuple
        for checkpoint_data in checkpoints:
            # Create config for aget_tuple
            checkpoint_config = {
                "configurable": {
                    "thread_id": checkpoint_data.get("thread_id"),
                    "checkpoint_id": checkpoint_data.get("id"),
                },
            }

            # Use aget_tuple to convert to CheckpointTuple
            tuple_result = await self.aget_tuple(checkpoint_config)
            if tuple_result:
                yield tuple_result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async version of put_writes method.

        Store pending writes for a checkpoint asynchronously.

        Args:
            config: The runnable configuration
            writes: Sequence of (channel, value) tuples to write
            task_id: Identifier for the task performing the writes
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id or not checkpoint_id:
            msg = "thread_id and checkpoint_id are required for put_writes"
            raise ConfigurationError(msg)

        # Store each pending write asynchronously
        for channel, value in writes:
            # Serialize the value
            if value is not None:
                data_bytes = self.serde.dumps(value)
                # Encode bytes to base64 for JSON storage
                serialized_value = base64.b64encode(data_bytes).decode("utf-8")
            else:
                serialized_value = None

            # Store in Qdrant asynchronously
            if self._pool:
                async with self._pool.acquire() as client:
                    await client.astore_pending_write(
                        checkpoint_id=checkpoint_id,
                        thread_id=thread_id,
                        task_id=task_id,
                        channel=channel,
                        value=serialized_value,
                    )
            else:
                await self._client.astore_pending_write(
                    checkpoint_id=checkpoint_id,
                    thread_id=thread_id,
                    task_id=task_id,
                    channel=channel,
                    value=serialized_value,
                )

    def search_checkpoints(
        self,
        query: str,
        thread_id: str | None = None,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> builtins.list[CheckpointTuple]:
        """Search for checkpoints using semantic similarity.

        Args:
            query: Search query text
            thread_id: Optional thread ID to filter by
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of CheckpointTuple objects sorted by relevance
        """
        # Generate embedding for query
        query_embedding = None
        if self._embedding_generator.embeddings:
            query_embedding = self._embedding_generator.embeddings.embed_query(query)

        if query_embedding is None:
            msg = "Embeddings not available for semantic search"
            raise ValueError(msg)

        # Search using client
        results = self._client.semantic_search(
            query_vector=query_embedding,
            thread_id=thread_id,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Convert results to CheckpointTuple objects
        checkpoints = []
        for result in results:
            try:
                # Extract data
                checkpoint_data = json.loads(result["checkpoint"])
                metadata_data = json.loads(result["metadata"])

                # Deserialize checkpoint
                try:
                    checkpoint = self.serde.loads_typed(("checkpoint", checkpoint_data))
                except Exception as e:
                    # Fallback to simple JSON deserialization for mocks/tests
                    if "Unknown serialization type" in str(e):
                        checkpoint = (
                            json.loads(checkpoint_data)
                            if isinstance(checkpoint_data, str)
                            else checkpoint_data
                        )
                    else:
                        raise

                try:
                    metadata = self.serde.loads_typed(("metadata", metadata_data))
                except Exception as e:
                    # Fallback to simple JSON deserialization for mocks/tests
                    if "Unknown serialization type" in str(e):
                        metadata = (
                            json.loads(metadata_data)
                            if isinstance(metadata_data, str)
                            else metadata_data
                        )
                    else:
                        raise

                # Create config
                config = {
                    "configurable": {
                        "thread_id": result["thread_id"],
                        "checkpoint_ns": result.get("checkpoint_ns", ""),
                        "checkpoint_id": result["checkpoint_id"],
                    },
                }

                # Create parent config if exists
                parent_config = None
                if result.get("parent_config"):
                    parent_config = {
                        "configurable": {
                            "thread_id": result["thread_id"],
                            "checkpoint_ns": result.get("checkpoint_ns", ""),
                            "checkpoint_id": result["parent_config"],
                        },
                    }

                # Create CheckpointTuple
                checkpoint_tuple = CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                    pending_writes=[],  # TODO: Load pending writes if needed
                )

                checkpoints.append(checkpoint_tuple)

            except Exception as e:
                logger.exception(f"Failed to deserialize checkpoint: {e}")
                continue

        return checkpoints

    async def asearch_checkpoints(
        self,
        query: str,
        thread_id: str | None = None,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> builtins.list[CheckpointTuple]:
        """Async version of search_checkpoints.

        Search for checkpoints using semantic similarity asynchronously.

        Args:
            query: Search query text
            thread_id: Optional thread ID to filter by
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of CheckpointTuple objects sorted by relevance
        """
        # For now, just call sync version
        # TODO: Make fully async when embedding generator supports it
        return await asyncio.to_thread(
            self.search_checkpoints,
            query,
            thread_id,
            limit,
            score_threshold,
        )

    def close(self) -> None:
        """Close the checkpointer and clean up resources.

        This method ensures that the connection pool is properly closed,
        which stops the health check loop and closes all connections.
        """
        if self._pool:
            # Close connection pool synchronously
            try:
                # Check if there's already a running event loop
                asyncio.get_running_loop()
                # If we're here, we're in an async context
                # Create a task to close the pool
                asyncio.create_task(self._pool.close())
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                asyncio.run(self._pool.close())

            self._pool = None
            logger.info("UnifiedCheckpointer closed successfully")

    async def aclose(self) -> None:
        """Async version of close.

        Close the checkpointer and clean up resources asynchronously.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("UnifiedCheckpointer closed successfully")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.close()
        return False  # Don't suppress exceptions

    async def __aenter__(self):
        """Async enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit context manager and clean up resources."""
        await self.aclose()
        return False  # Don't suppress exceptions
