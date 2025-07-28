"""Basic usage example for LangChain Memory Integration.

This example shows the simplest way to use the EnhancedMemoryCheckpointer
for saving and loading conversation memory.
"""
from langchain_memory_integration import EnhancedMemoryCheckpointer

def main():
    # Initialize the checkpointer
    checkpointer = EnhancedMemoryCheckpointer()
    
    # Save a conversation to memory
    memory_id = checkpointer.save_sync({
        "conversation": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great! How can I help you today?"},
            {"role": "user", "content": "Can you explain quantum computing?"},
            {"role": "assistant", "content": "Quantum computing uses quantum bits..."}
        ],
        "metadata": {
            "topic": "quantum computing",
            "timestamp": "2025-07-28T07:30:00Z"
        }
    })
    print(f"Saved memory with ID: {memory_id}")
    
    # Load the memory back
    loaded_memory = checkpointer.load_sync(memory_id)
    print(f"Loaded memory: {loaded_memory}")
    
    # Search for memories by content
    search_results = checkpointer.search_sync("quantum", limit=5)
    print(f"Found {len(search_results)} memories about quantum topics")
    
    # List recent memories
    recent = checkpointer.list_memories_sync(limit=10)
    print(f"Total recent memories: {len(recent)}")

if __name__ == "__main__":
    main()
