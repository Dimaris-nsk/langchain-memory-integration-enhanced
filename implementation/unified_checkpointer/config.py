"""Configuration for UnifiedCheckpointer."""

from dataclasses import dataclass

from langgraph.checkpoint.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


@dataclass
class UnifiedCheckpointerConfig:
    """Configuration for UnifiedCheckpointer.

    Attributes:
        collection_name: Name of the unified-memory collection
        unified_memory_url: URL for unified-memory service (optional)
        api_key: API key for unified-memory (optional)
        cache_size: Maximum number of checkpoints to cache
        connection_pool_size: Number of connections in the pool
        batch_size: Size for batch operations
        enable_search: Enable semantic search functionality
        enable_analytics: Enable analytics features
        auto_summarize: Automatically generate summaries for checkpoints
        retry_attempts: Number of retry attempts for failed operations
        circuit_breaker_threshold: Failure count before circuit opens
        fallback_to_memory: Use in-memory fallback on failures
        serializer: Serialization protocol to use
        compress_large_states: Enable compression for large states
        compression_threshold_kb: Size threshold for compression
    """

    # Connection settings
    collection_name: str = "langgraph_checkpoints"
    unified_memory_url: str | None = None
    api_key: str | None = None

    # Performance tuning
    cache_size: int = 1000
    connection_pool_size: int = 10
    batch_size: int = 100

    # Features
    enable_search: bool = True
    enable_analytics: bool = True
    auto_summarize: bool = True

    # Resilience
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    fallback_to_memory: bool = True

    # Serialization
    serializer: SerializerProtocol = None
    compress_large_states: bool = True
    compression_threshold_kb: int = 100

    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600  # 1 hour

    # Connection pooling
    pool_enabled: bool = True
    pool_min_connections: int = 2
    pool_max_connections: int = 10
    pool_connection_ttl: float = 3600.0  # 1 hour
    pool_idle_timeout: float = 600.0  # 10 minutes
    pool_health_check_interval: float = 60.0  # 1 minute

    def __post_init__(self):
        """Initialize defaults after dataclass initialization."""
        if self.serializer is None:
            self.serializer = JsonPlusSerializer()
