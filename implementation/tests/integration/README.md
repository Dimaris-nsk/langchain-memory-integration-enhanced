# Integration Tests

This directory contains integration tests for UnifiedCheckpointer with LangGraph.

## Test Coverage

### Basic Integration (`test_langgraph_integration.py`)
- ✅ Basic StateGraph compilation with checkpointer
- ✅ State persistence across invocations
- ✅ Thread isolation
- ✅ Messages state with add_messages reducer
- ✅ Multi-turn conversations
- ✅ Checkpoint history retrieval
- ✅ Resume from checkpoint / time travel
- ✅ State updates
- ✅ Async execution
- ✅ Async streaming
- ✅ Pending writes handling
- ✅ Error recovery

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run specific test class
pytest tests/integration/test_langgraph_integration.py::TestBasicIntegration

# Run with verbose output
pytest -v tests/integration/

# Run async tests
pytest -v tests/integration/ -k "async"
```

## Requirements

- LangGraph >= 0.5.4
- langchain-core
- pytest-asyncio (for async tests)
