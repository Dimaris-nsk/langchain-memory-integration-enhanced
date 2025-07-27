# LangChain Memory Integration Enhanced - API Reference

## 📖 Overview

This document provides a comprehensive API reference for the LangChain Memory Integration Enhanced library. All examples assume the following imports:

```python
from langchain_memory_integration import (
    EnhancedMemoryCheckpointer,
    Settings,
    VectorStoreType,
    HealthStatus
)
```

## 🔧 Configuration

### Settings Class

The `Settings` class manages all configuration options for the library.

```python
from langchain_memory_integration.config import Settings

# Create settings with environment variables
settings = Settings()

# Create settings with explicit values
settings = Settings(
    vector_store_type=VectorStoreType.QDRANT,
    qdrant_url="localhost:6333",
    qdrant_collection="memories",
    cache_enabled=True,
    redis_url="redis://localhost:6379"
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_store_type` | `VectorStoreType` | `QDRANT` | Vector store backend |
| `qdrant_url` | `str` | `"localhost:6333"` | Qdrant server URL |
| `qdrant_collection` | `str` | `"langchain_memory"` | Qdrant collection name |
| `qdrant_api_key` | `Optional[str]` | `None` | Qdrant API key |
| `chroma_persist_dir` | `str` | `"./chroma_db"` | Chroma persistence directory |
| `pinecone_api_key` | `Optional[str]` | `None` | Pinecone API key |
| `pinecone_environment` | `Optional[str]` | `None` | Pinecone environment |
| `weaviate_url` | `Optional[str]` | `None` | Weaviate server URL |
| `cache_enabled` | `bool` | `False` | Enable caching |
| `redis_url` | `Optional[str]` | `None` | Redis URL for caching |
| `batch_size` | `int` | `100` | Batch operation size |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `retry_delay` | `float` | `1.0` | Base retry delay (seconds) |

## 🎯 Core API

### EnhancedMemoryCheckpointer

The main interface for memory operations.

#### Initialization

```python
# Basic initialization
checkpointer = EnhancedMemoryCheckpointer()

# With custom settings
settings = Settings(vector_store_type=VectorStoreType.CHROMA)
checkpointer = EnhancedMemoryCheckpointer(settings=settings)

# With existing LangChain checkpointer
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
checkpointer = EnhancedMemoryCheckpointer(
    settings=settings,
    base_checkpointer=memory
)
```

#### Save Memory

```python
# Save memory state
memory_id = await checkpointer.save(
    memory_data={
        "conversation": [...],
        "metadata": {...}
    },
    tags=["session-123", "user-456"],
    metadata={
        "timestamp": "2025-07-28T12:00:00Z",
        "version": "1.0"
    }
)

# Sync version
memory_id = checkpointer.save_sync(
    memory_data={...},
    tags=[...],
    metadata={...}
)
```

#### Load Memory

```python
# Load by ID
memory_data = await checkpointer.load(memory_id="abc123")

# Load with filters
memories = await checkpointer.load(
    tags=["session-123"],
    metadata_filter={"version": "1.0"},
    limit=10
)

# Sync version
memory_data = checkpointer.load_sync(memory_id="abc123")
```

#### Search Memory

```python
# Semantic search
results = await checkpointer.search(
    query="discussions about machine learning",
    limit=5,
    score_threshold=0.7
)

# Search with filters
results = await checkpointer.search(
    query="project planning",
    tags=["team-alpha"],
    metadata_filter={"status": "active"},
    limit=10
)
```

#### Update Memory

```python
# Update existing memory
success = await checkpointer.update(
    memory_id="abc123",
    memory_data={...},
    tags=["updated"],
    metadata={"last_modified": "2025-07-28"}
)
```

#### Delete Memory

```python
# Delete by ID
success = await checkpointer.delete(memory_id="abc123")

# Delete by filter
deleted_count = await checkpointer.delete(
    tags=["obsolete"],
    metadata_filter={"created_before": "2025-01-01"}
)
```

#### Batch Operations

```python
# Batch save
memory_ids = await checkpointer.batch_save([
    {
        "memory_data": {...},
        "tags": ["batch-1"],
        "metadata": {...}
    },
    {
        "memory_data": {...},
        "tags": ["batch-2"],
        "metadata": {...}
    }
])

# Batch load
memories = await checkpointer.batch_load(
    memory_ids=["abc123", "def456", "ghi789"]
)

# Batch delete
success = await checkpointer.batch_delete(
    memory_ids=["abc123", "def456"]
)
```

## 🔄 Migration API

### MigrationEngine

Handles migration between different storage backends.

```python
from langchain_memory_integration.migration_tools import MigrationEngine

# Create migration engine
engine = MigrationEngine(
    source_settings=Settings(vector_store_type=VectorStoreType.QDRANT),
    target_settings=Settings(vector_store_type=VectorStoreType.CHROMA)
)

# Run migration
result = await engine.migrate(
    batch_size=1000,
    parallel_workers=4,
    progress_callback=lambda p: print(f"Progress: {p}%")
)

# Verify migration
verification = await engine.verify_migration()
```

## 🛡️ Resilience API

### Circuit Breaker

```python
from langchain_memory_integration.resilience import CircuitBreaker

# Get circuit breaker instance
breaker = checkpointer.circuit_breaker

# Check status
status = breaker.status  # CLOSED, OPEN, or HALF_OPEN

# Manual control
breaker.reset()  # Reset to CLOSED state
breaker.force_open()  # Force OPEN state
```

### Health Check

```python
# Check system health
health = await checkpointer.health_check()

print(f"Status: {health.status}")  # HEALTHY, DEGRADED, or UNHEALTHY
print(f"Details: {health.details}")
print(f"Checks: {health.checks}")

# Individual component health
storage_health = await checkpointer.check_storage_health()
cache_health = await checkpointer.check_cache_health()
```

## 🔗 Compatibility API

### Version Adapter

```python
from langchain_memory_integration.compatibility import VersionAdapter

# Auto-detect version
adapter = VersionAdapter.auto_detect()

# Specific version
adapter = VersionAdapter.for_version("0.2.5")

# Check compatibility
is_compatible = adapter.is_compatible_with("0.3.0")
```

## 🧰 Utility APIs

### Serialization

```python
from langchain_memory_integration.utils import (
    JsonSerializer,
    PickleSerializer,
    MessagePackSerializer
)

# Use custom serializer
checkpointer = EnhancedMemoryCheckpointer(
    serializer=MessagePackSerializer()
)
```

### Logging

```python
import logging
from langchain_memory_integration import configure_logging

# Configure library logging
configure_logging(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Metrics

```python
# Get operation metrics
metrics = checkpointer.get_metrics()

print(f"Total saves: {metrics.total_saves}")
print(f"Total loads: {metrics.total_loads}")
print(f"Cache hit rate: {metrics.cache_hit_rate}")
print(f"Average latency: {metrics.avg_latency_ms}ms")
```

## 🔌 Extension APIs

### Custom Storage Backend

```python
from langchain_memory_integration.core import BaseVectorStore

class CustomVectorStore(BaseVectorStore):
    async def save(self, data: Dict, **kwargs) -> str:
        # Implementation
        pass
    
    async def load(self, id: str, **kwargs) -> Dict:
        # Implementation
        pass
    
    async def search(self, query: str, limit: int, **kwargs) -> List[Dict]:
        # Implementation
        pass
    
    async def delete(self, id: str) -> bool:
        # Implementation
        pass

# Register custom backend
from langchain_memory_integration import register_backend

register_backend("custom", CustomVectorStore)
```

### Custom Health Check

```python
from langchain_memory_integration.health import BaseHealthCheck

class CustomHealthCheck(BaseHealthCheck):
    async def check(self) -> HealthStatus:
        # Implementation
        return HealthStatus(
            status="HEALTHY",
            details={"custom": "check passed"}
        )

# Add to checkpointer
checkpointer.add_health_check("custom", CustomHealthCheck())
```

## 🌐 Environment Variables

The library supports configuration through environment variables:

```bash
# Vector store configuration
LANGCHAIN_VECTOR_STORE_TYPE=qdrant
LANGCHAIN_QDRANT_URL=http://localhost:6333
LANGCHAIN_QDRANT_COLLECTION=my_memories
LANGCHAIN_QDRANT_API_KEY=your-api-key

# Cache configuration
LANGCHAIN_CACHE_ENABLED=true
LANGCHAIN_REDIS_URL=redis://localhost:6379/0

# Resilience configuration
LANGCHAIN_MAX_RETRIES=3
LANGCHAIN_RETRY_DELAY=1.0
LANGCHAIN_CIRCUIT_BREAKER_THRESHOLD=5
LANGCHAIN_CIRCUIT_BREAKER_TIMEOUT=60

# Performance configuration
LANGCHAIN_BATCH_SIZE=100
LANGCHAIN_CONNECTION_POOL_SIZE=10
```

## 📝 Type Definitions

### VectorStoreType

```python
from enum import Enum

class VectorStoreType(str, Enum):
    QDRANT = "qdrant"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
```

### HealthStatus

```python
from typing import Dict, Any, Literal

class HealthStatus:
    status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
    details: Dict[str, Any]
    checks: Dict[str, bool]
    timestamp: str
```

### MemoryData

```python
from typing import Dict, List, Any, Optional

class MemoryData:
    id: str
    data: Dict[str, Any]
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: Optional[str]
```

## 🚨 Exception Handling

### Custom Exceptions

```python
from langchain_memory_integration.exceptions import (
    MemoryNotFoundError,
    StorageConnectionError,
    SerializationError,
    MigrationError,
    CircuitBreakerOpenError
)

try:
    memory = await checkpointer.load("invalid-id")
except MemoryNotFoundError as e:
    print(f"Memory not found: {e}")
except StorageConnectionError as e:
    print(f"Storage error: {e}")
```

### Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| `MEM001` | `MemoryNotFoundError` | Memory ID not found |
| `MEM002` | `InvalidMemoryDataError` | Invalid memory data format |
| `STO001` | `StorageConnectionError` | Cannot connect to storage |
| `STO002` | `StorageTimeoutError` | Storage operation timeout |
| `SER001` | `SerializationError` | Serialization failed |
| `MIG001` | `MigrationError` | Migration failed |
| `RES001` | `CircuitBreakerOpenError` | Circuit breaker is open |

## 📚 Examples

### Complete Example

```python
import asyncio
from langchain_memory_integration import (
    EnhancedMemoryCheckpointer,
    Settings,
    VectorStoreType
)

async def main():
    # Configure
    settings = Settings(
        vector_store_type=VectorStoreType.QDRANT,
        cache_enabled=True,
        redis_url="redis://localhost:6379"
    )
    
    # Initialize
    checkpointer = EnhancedMemoryCheckpointer(settings)
    
    # Save memory
    memory_id = await checkpointer.save(
        memory_data={
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        tags=["greeting", "session-123"],
        metadata={"user_id": "user-456"}
    )
    
    # Search memories
    results = await checkpointer.search(
        query="greeting conversation",
        limit=5
    )
    
    # Load specific memory
    memory = await checkpointer.load(memory_id)
    
    # Update memory
    await checkpointer.update(
        memory_id=memory_id,
        memory_data={
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }
    )
    
    # Health check
    health = await checkpointer.health_check()
    print(f"System health: {health.status}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

*Last updated: 2025-07-28*