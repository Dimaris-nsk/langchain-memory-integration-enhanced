"""Migration example for LangChain Memory Integration.

This example shows how to migrate from standard LangChain memory
to the enhanced memory integration system.
"""
from langchain.memory import ConversationBufferMemory
from langchain_memory_integration import EnhancedMemoryCheckpointer
from langchain_memory_integration.migration_tools import MemoryMigrator

def migrate_simple_memory():
    """Migrate from ConversationBufferMemory."""
    # Original LangChain memory
    old_memory = ConversationBufferMemory()
    old_memory.save_context(
        {"input": "Hello, my name is Alice"},
        {"output": "Nice to meet you, Alice!"}
    )
    old_memory.save_context(
        {"input": "What's my name?"},
        {"output": "Your name is Alice."}
    )
    
    # Migrate to enhanced memory
    migrator = MemoryMigrator()
    checkpointer = EnhancedMemoryCheckpointer()
    
    # Convert memory format
    enhanced_data = migrator.convert_buffer_memory(old_memory)
    
    # Save to enhanced system
    memory_id = checkpointer.save_sync(enhanced_data)
    print(f"Migrated memory saved with ID: {memory_id}")
    
    # Verify migration
    loaded = checkpointer.load_sync(memory_id)
    print(f"Migrated conversations: {len(loaded['conversation'])} messages")

def migrate_with_metadata():
    """Migrate with custom metadata preservation."""
    from langchain.memory import ConversationSummaryMemory
    from langchain.llms import OpenAI
    
    # Original summary memory
    llm = OpenAI(temperature=0)
    old_memory = ConversationSummaryMemory(llm=llm)
    old_memory.save_context(
        {"input": "Tell me about machine learning"},
        {"output": "Machine learning is a subset of AI..."}
    )
    
    # Migrate with metadata
    migrator = MemoryMigrator()
    enhanced_data = migrator.convert_summary_memory(
        old_memory,
        metadata={
            "source": "customer_support",
            "session_id": "sess_123",
            "migrated_at": "2025-07-28T07:35:00Z"
        }
    )
    
    checkpointer = EnhancedMemoryCheckpointer()
    memory_id = checkpointer.save_sync(enhanced_data)
    print(f"Migrated with metadata: {memory_id}")

def batch_migration_example():
    """Migrate multiple memories at once."""
    # Multiple old memories
    memories = []
    for i in range(3):
        mem = ConversationBufferMemory()
        mem.save_context(
            {"input": f"User message {i}"},
            {"output": f"Assistant response {i}"}
        )
        memories.append(mem)
    
    # Batch migration
    migrator = MemoryMigrator()
    checkpointer = EnhancedMemoryCheckpointer()
    
    migrated_ids = []
    for i, old_mem in enumerate(memories):
        enhanced_data = migrator.convert_buffer_memory(
            old_mem,
            metadata={"batch_index": i, "batch_id": "migration_001"}
        )
        memory_id = checkpointer.save_sync(enhanced_data)
        migrated_ids.append(memory_id)
    
    print(f"Batch migrated {len(migrated_ids)} memories")
    
    # Verify batch migration
    batch_search = checkpointer.search_sync(
        query="",
        filters={"metadata.batch_id": "migration_001"},
        limit=10
    )
    print(f"Found {len(batch_search)} migrated memories in batch")

def main():
    """Run migration examples."""
    print("=== Simple Migration ===")
    migrate_simple_memory()
    
    print("\n=== Migration with Metadata ===")
    migrate_with_metadata()
    
    print("\n=== Batch Migration ===")
    batch_migration_example()

if __name__ == "__main__":
    main()
