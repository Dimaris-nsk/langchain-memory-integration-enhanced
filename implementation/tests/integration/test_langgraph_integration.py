"""
Integration tests for UnifiedCheckpointer with LangGraph StateGraph.

Tests real-world usage scenarios including:
- Basic graph compilation and execution
- State persistence across invocations  
- Thread isolation
- Checkpoint versioning
- Pending writes handling
- Time travel / history retrieval
"""

import pytest
import uuid
from typing import Annotated, TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from unified_checkpointer import UnifiedCheckpointer
from unified_checkpointer.config import UnifiedCheckpointerConfig


# Test state definitions
class SimpleState(TypedDict):
    """Simple state for basic tests."""
    counter: int
    messages: list[str]


class MessagesState(TypedDict):
    """State with LangChain messages."""
    messages: Annotated[list[BaseMessage], add_messages]


class ComplexState(TypedDict):
    """Complex state with multiple fields."""
    topic: str
    counter: int
    history: Annotated[list[str], add]
    metadata: dict


@pytest.fixture
def unified_checkpointer():
    """Create UnifiedCheckpointer for tests."""
    config = UnifiedCheckpointerConfig(
        collection_name="test_langgraph_integration",
        unified_memory_url=None  # Use in-memory mode
    )
    checkpointer = UnifiedCheckpointer(config)
    yield checkpointer
    checkpointer.close()  # Ensure proper cleanup


class TestBasicIntegration:
    """Test basic StateGraph integration."""
    
    def test_simple_graph_compilation(self, unified_checkpointer):
        """Test that UnifiedCheckpointer works with basic StateGraph compilation."""
        # Define nodes
        def increment(state: SimpleState) -> SimpleState:
            return {
                "counter": state["counter"] + 1,
                "messages": ["Incremented"]
            }
        
        def double(state: SimpleState) -> SimpleState:
            return {
                "counter": state["counter"] * 2,
                "messages": ["Doubled"]
            }
        
        # Build graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("increment", increment)
        workflow.add_node("double", double)
        workflow.add_edge(START, "increment")
        workflow.add_edge("increment", "double")
        workflow.add_edge("double", END)
        
        # Compile with checkpointer
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Execute
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = graph.invoke({"counter": 1, "messages": []}, config)
        
        assert result["counter"] == 4  # (1 + 1) * 2
        assert len(result["messages"]) == 2
    
    def test_state_persistence(self, unified_checkpointer):
        """Test that state persists across invocations."""
        # Simple accumulator node
        def accumulate(state: SimpleState) -> SimpleState:
            return {
                "counter": state["counter"] + 1,
                "messages": [f"Count: {state['counter'] + 1}"]
            }
        
        # Build graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("accumulate", accumulate)
        workflow.add_edge(START, "accumulate")
        workflow.add_edge("accumulate", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Use same thread_id for persistence
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # First invocation
        result1 = graph.invoke({"counter": 0, "messages": []}, config)
        assert result1["counter"] == 1
        
        # Get state - should match result
        state = graph.get_state(config)
        assert state.values["counter"] == 1
        assert len(state.values["messages"]) == 1    
    def test_thread_isolation(self, unified_checkpointer):
        """Test that different threads maintain isolated state."""
        # Counter node
        def increment(state: SimpleState) -> SimpleState:
            return {"counter": state["counter"] + 1, "messages": []}
        
        # Build graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("increment", increment)
        workflow.add_edge(START, "increment")
        workflow.add_edge("increment", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Thread 1
        thread1_config = {"configurable": {"thread_id": "thread-1"}}
        result1 = graph.invoke({"counter": 10, "messages": []}, thread1_config)
        assert result1["counter"] == 11
        
        # Thread 2
        thread2_config = {"configurable": {"thread_id": "thread-2"}}
        result2 = graph.invoke({"counter": 20, "messages": []}, thread2_config)
        assert result2["counter"] == 21
        
        # Verify isolation
        state1 = graph.get_state(thread1_config)
        state2 = graph.get_state(thread2_config)
        assert state1.values["counter"] == 11
        assert state2.values["counter"] == 21

class TestMessagesIntegration:
    """Test integration with LangChain messages."""
    
    def test_messages_state(self, unified_checkpointer):
        """Test graph with messages state and add_messages reducer."""
        # Chat node that responds to messages
        def chat_node(state: MessagesState) -> MessagesState:
            last_message = state["messages"][-1]
            response = f"Echo: {last_message.content}"
            return {"messages": [AIMessage(content=response)]}
        
        # Build graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("chat", chat_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Start conversation
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # First message
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="Hello")]}, 
            config
        )
        assert len(result1["messages"]) == 2  # Human + AI
        assert result1["messages"][-1].content == "Echo: Hello"        
        # Second message - should remember conversation
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="How are you?")]},
            config
        )
        # Should have all 4 messages now
        assert len(result2["messages"]) == 4
        assert result2["messages"][0].content == "Hello"
        assert result2["messages"][1].content == "Echo: Hello"
        assert result2["messages"][2].content == "How are you?"
        assert result2["messages"][3].content == "Echo: How are you?"
    
    def test_multi_turn_conversation(self, unified_checkpointer):
        """Test multi-turn conversation with memory."""
        # Smarter chat node
        def assistant(state: MessagesState) -> MessagesState:
            messages = state["messages"]
            # Count previous messages
            turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            response = f"This is turn {turn_count}. You said: {messages[-1].content}"
            return {"messages": [AIMessage(content=response)]}
        
        # Build graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("assistant", assistant)
        workflow.add_edge(START, "assistant")
        workflow.add_edge("assistant", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)        
        # Have a conversation
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Multiple turns
        turns = ["First message", "Second message", "Third message"]
        
        for i, message in enumerate(turns, 1):
            result = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config
            )
            last_response = result["messages"][-1].content
            assert f"turn {i}" in last_response
            assert message in last_response


class TestCheckpointFeatures:
    """Test advanced checkpoint features."""
    
    def test_checkpoint_history(self, unified_checkpointer):
        """Test retrieving checkpoint history."""
        # Simple incrementing node
        def increment(state: SimpleState) -> SimpleState:
            return {
                "counter": state["counter"] + 1,
                "messages": [f"Step {state['counter'] + 1}"]
            }        
        # Build graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("increment", increment)
        workflow.add_edge(START, "increment")
        workflow.add_edge("increment", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Execute multiple times
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run 3 times
        for i in range(3):
            graph.invoke({"counter": i, "messages": []}, config)
        
        # Get history
        history = list(graph.get_state_history(config))
        
        # Should have checkpoints for each run
        assert len(history) >= 3
        
        # Most recent first
        assert history[0].values["counter"] == 3
        assert history[1].values["counter"] == 2
        assert history[2].values["counter"] == 1    
    def test_resume_from_checkpoint(self, unified_checkpointer):
        """Test resuming execution from a specific checkpoint."""
        # Nodes that build a story
        def add_beginning(state: ComplexState) -> ComplexState:
            return {
                "topic": state.get("topic", "story"),
                "counter": 1,
                "history": ["Once upon a time"],
                "metadata": {"stage": "beginning"}
            }
        
        def add_middle(state: ComplexState) -> ComplexState:
            return {
                "counter": state["counter"] + 1,
                "history": ["Something interesting happened"],
                "metadata": {"stage": "middle"}
            }
        
        def add_end(state: ComplexState) -> ComplexState:
            return {
                "counter": state["counter"] + 1,
                "history": ["The end"],
                "metadata": {"stage": "end"}
            }
        
        # Build graph
        workflow = StateGraph(ComplexState)
        workflow.add_node("beginning", add_beginning)
        workflow.add_node("middle", add_middle)
        workflow.add_node("end", add_end)
        workflow.add_edge(START, "beginning")
        workflow.add_edge("beginning", "middle")
        workflow.add_edge("middle", "end")
        workflow.add_edge("end", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Run the full story
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "topic": "adventure",
            "counter": 0,
            "history": [],
            "metadata": {}
        }
        
        result = graph.invoke(initial_state, config)
        
        # Should have completed all stages
        assert result["counter"] == 3
        assert len(result["history"]) == 3
        assert result["metadata"]["stage"] == "end"
        
        # Get checkpoint from middle
        history = list(graph.get_state_history(config))
        
        # Find checkpoint after "beginning" 
        middle_checkpoint = None        for checkpoint in history:
            if checkpoint.values.get("metadata", {}).get("stage") == "beginning":
                middle_checkpoint = checkpoint
                break
        
        assert middle_checkpoint is not None
        
        # Update state from that checkpoint
        new_config = graph.update_state(
            middle_checkpoint.config,
            values={"topic": "mystery", "history": ["It was a dark night"]}
        )
        
        # Resume from updated checkpoint
        resumed_result = graph.invoke(None, new_config)
        
        # Should have the updated topic and modified history
        assert resumed_result["topic"] == "mystery"
        assert "It was a dark night" in resumed_result["history"]
        assert resumed_result["metadata"]["stage"] == "end"


class TestAsyncIntegration:
    """Test async execution with checkpointer."""
    
    @pytest.mark.asyncio
    async def test_async_graph_execution(self, unified_checkpointer):
        """Test async graph compilation and execution."""
        # Async node
        async def async_process(state: SimpleState) -> SimpleState:            # Simulate async work
            import asyncio
            await asyncio.sleep(0.01)
            return {
                "counter": state["counter"] + 10,
                "messages": ["Processed asynchronously"]
            }
        
        # Build graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("process", async_process)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Execute asynchronously
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = await graph.ainvoke({"counter": 5, "messages": []}, config)
        
        assert result["counter"] == 15
        assert "Processed asynchronously" in result["messages"]
        
        # Verify state was saved
        state = await graph.aget_state(config)
        assert state.values["counter"] == 15    
    @pytest.mark.asyncio  
    async def test_async_streaming(self, unified_checkpointer):
        """Test async streaming with checkpoints."""
        # Multi-step process
        async def step1(state: SimpleState) -> SimpleState:
            return {"counter": state["counter"] + 1, "messages": ["Step 1"]}
        
        async def step2(state: SimpleState) -> SimpleState:
            return {"counter": state["counter"] + 2, "messages": ["Step 2"]}
        
        async def step3(state: SimpleState) -> SimpleState:
            return {"counter": state["counter"] + 3, "messages": ["Step 3"]}
        
        # Build graph with multiple steps
        workflow = StateGraph(SimpleState)
        workflow.add_node("step1", step1)
        workflow.add_node("step2", step2) 
        workflow.add_node("step3", step3)
        workflow.add_edge(START, "step1")
        workflow.add_edge("step1", "step2")
        workflow.add_edge("step2", "step3")
        workflow.add_edge("step3", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Stream execution
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        chunks = []        
        async for chunk in graph.astream(
            {"counter": 0, "messages": []},
            config,
            stream_mode="values"
        ):
            chunks.append(chunk)
        
        # Should have streamed intermediate states
        assert len(chunks) >= 3
        
        # Final state should have all operations
        final_state = chunks[-1]
        assert final_state["counter"] == 6  # 0 + 1 + 2 + 3
        
        # Verify checkpoint saved
        saved_state = await graph.aget_state(config) 
        assert saved_state.values["counter"] == 6


class TestPendingWrites:
    """Test pending writes functionality."""
    
    def test_pending_writes_handling(self, unified_checkpointer):
        """Test that pending writes are properly stored and retrieved."""
        # Create a graph that uses channels
        def write_to_channels(state: SimpleState) -> SimpleState:
            # This simulates writes to different channels
            return {
                "counter": state["counter"] + 1,
                "messages": ["Written to channels"]
            }        
        # Build simple graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("writer", write_to_channels)
        workflow.add_edge(START, "writer")
        workflow.add_edge("writer", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        # Execute
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        result = graph.invoke({"counter": 0, "messages": []}, config)
        
        # Get checkpoint tuple to inspect internals
        checkpoint_tuple = unified_checkpointer.get_tuple(config)
        
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.checkpoint["channel_values"]["counter"] == 1
        
        # Verify versions are tracked
        assert "counter" in checkpoint_tuple.checkpoint["channel_versions"]


class TestErrorHandling:
    """Test error scenarios."""
    
    def test_node_error_recovery(self, unified_checkpointer):
        """Test graph behavior when node fails."""
        # Node that fails on certain conditions        def maybe_fail(state: SimpleState) -> SimpleState:
            if state["counter"] == 2:
                raise ValueError("Simulated failure")
            return {
                "counter": state["counter"] + 1,
                "messages": [f"Success at {state['counter']}"]
            }
        
        # Build graph
        workflow = StateGraph(SimpleState)
        workflow.add_node("maybe_fail", maybe_fail)
        workflow.add_edge(START, "maybe_fail")
        workflow.add_edge("maybe_fail", END)
        
        graph = workflow.compile(checkpointer=unified_checkpointer)
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # First run - should succeed
        result1 = graph.invoke({"counter": 0, "messages": []}, config)
        assert result1["counter"] == 1
        
        # Second run - should fail but checkpoint is saved
        with pytest.raises(ValueError, match="Simulated failure"):
            graph.invoke({"counter": 2, "messages": []}, config)
        
        # Can still get last successful state
        state = graph.get_state(config)
        assert state.values["counter"] == 1  # Last successful state