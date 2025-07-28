"""Advanced features example for LangChain Memory Integration.

This example demonstrates advanced features including:
- Async operations
- Batch processing
- Custom metadata
- Error handling and resilience
- Performance optimization
"""
import asyncio
from langchain_memory_integration import (
    EnhancedMemoryCheckpointer,
    MemoryError,
    MemoryNotFoundError
)

async def demonstrate_async_operations():
    """Show async save/load operations."""
    checkpointer = EnhancedMemoryCheckpointer()
    
    # Async save
    memory_id = await checkpointer.save_async({
        "conversation": [
            {"role": "user", "content": "Let's discuss AI ethics"},
            {"role": "assistant", "content": "AI ethics is crucial..."}
        ],
        "metadata": {
            "topic": "AI ethics",
            "importance": "high",
            "tags": ["ethics", "AI", "philosophy"]
        }
    })
    print(f"Async saved: {memory_id}")
    
    # Async load with error handling
    try:
        memory = await checkpointer.load_async(memory_id)
        print(f"Async loaded: {memory}")
    except MemoryNotFoundError:
        print("Memory not found!")
    except MemoryError as e:
        print(f"Memory error: {e}")

async def demonstrate_batch_operations():
    """Show batch processing capabilities."""
    checkpointer = EnhancedMemoryCheckpointer()
    
    # Prepare multiple memories
    memories = [
        {
            "conversation": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"}
            ],
            "metadata": {"index": i, "batch": "demo"}
        }
        for i in range(5)
    ]
    
    # Batch save
    memory_ids = await checkpointer.batch_save_async(memories)
    print(f"Batch saved {len(memory_ids)} memories")
    
    # Batch load
    loaded_memories = await checkpointer.batch_load_async(memory_ids)
    print(f"Batch loaded {len(loaded_memories)} memories")

def demonstrate_search_features():
    """Show advanced search capabilities."""
    checkpointer = EnhancedMemoryCheckpointer()
    
    # Search with filters
    results = checkpointer.search_sync(
        query="AI ethics",
        filters={
            "metadata.importance": "high",
            "metadata.tags": {"$contains": "ethics"}
        },
        limit=10
    )
    print(f"Filtered search found {len(results)} results")
    
    # Search with similarity threshold
    similar = checkpointer.search_sync(
        query="artificial intelligence morality",
        similarity_threshold=0.8,
        limit=5
    )
    print(f"High similarity search found {len(similar)} results")

def demonstrate_resilience():
    """Show resilience and error recovery."""
    from langchain_memory_integration.resilience import RetryConfig
    
    # Configure custom retry behavior
    checkpointer = EnhancedMemoryCheckpointer(
        retry_config=RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=10.0,
            exponential_base=2.0
        )
    )
    
    # Operations will automatically retry on failure
    try:
        memory_id = checkpointer.save_sync({
            "data": "important",
            "metadata": {"retry_demo": True}
        })
        print(f"Saved with retry support: {memory_id}")
    except Exception as e:
        print(f"Failed after retries: {e}")

async def main():
    """Run all demonstrations."""
    print("=== Async Operations ===")
    await demonstrate_async_operations()
    
    print("\n=== Batch Operations ===")
    await demonstrate_batch_operations()
    
    print("\n=== Search Features ===")
    demonstrate_search_features()
    
    print("\n=== Resilience ===")
    demonstrate_resilience()

if __name__ == "__main__":
    asyncio.run(main())
