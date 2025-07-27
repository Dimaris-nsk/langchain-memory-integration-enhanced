"""
Performance benchmarks for UnifiedCheckpointer.

Tests performance characteristics including:
- Latency for individual operations
- Throughput under load
- Scalability with checkpoint count
- Concurrent access performance
- Memory usage patterns
"""

import asyncio
import concurrent.futures
import time
import uuid
from datetime import datetime

import pytest
from langgraph.checkpoint.base import Checkpoint
from unified_checkpointer import UnifiedCheckpointer
from unified_checkpointer.config import UnifiedCheckpointerConfig


@pytest.fixture
def checkpointer():
    """Create UnifiedCheckpointer for benchmarks."""
    config = UnifiedCheckpointerConfig(
        collection_name="test_performance",
        unified_memory_url=None,  # Use in-memory mode
    )
    checkpointer = UnifiedCheckpointer(config)
    yield checkpointer
    checkpointer.close()  # Ensure proper cleanup


@pytest.fixture
def sample_checkpoint() -> Checkpoint:
    """Create a sample checkpoint for testing."""
    return {
        "v": 4,
        "ts": datetime.utcnow().isoformat(),
        "id": str(uuid.uuid4()),
        "channel_values": {
            "messages": ["Hello", "World"],
            "counter": 42,
            "data": {"key": "value" * 100},  # Some bulk data
        },
        "channel_versions": {
            "__start__": 1,
            "messages": 2,
            "counter": 3,
            "data": 4,
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
        },
    }


class TestLatency:
    """Test latency of individual operations."""

    def test_put_latency(self, checkpointer, sample_checkpoint, benchmark) -> None:
        """Benchmark put operation latency."""
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        def put_checkpoint() -> None:
            checkpointer.put(
                config=config,
                checkpoint=sample_checkpoint,
                metadata={"source": "benchmark"},
                new_versions={},
            )

        # pytest-benchmark will measure this
        result = benchmark(put_checkpoint)
        assert result is None  # put returns None
    def test_get_tuple_latency(self, checkpointer, sample_checkpoint, benchmark) -> None:
        """Benchmark get_tuple operation latency."""
        # Setup - put a checkpoint first
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        checkpointer.put(config, sample_checkpoint, {}, {})

        def get_checkpoint():
            return checkpointer.get_tuple(config)

        result = benchmark(get_checkpoint)
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    def test_list_latency(self, checkpointer, benchmark) -> None:
        """Benchmark list operation latency."""
        # Setup - create multiple checkpoints
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        for i in range(10):
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": str(uuid.uuid4()),
                "channel_values": {"counter": i},
                "channel_versions": {"counter": i + 1},
                "versions_seen": {},
            }
            checkpointer.put(config, checkpoint, {"step": i}, {})

        def list_checkpoints():
            return list(checkpointer.list(config))

        results = benchmark(list_checkpoints)
        assert len(results) == 10

class TestThroughput:
    """Test throughput under load."""

    def test_put_throughput(self, checkpointer, sample_checkpoint) -> None:
        """Test how many puts per second we can handle."""
        num_operations = 1000
        thread_id = str(uuid.uuid4())

        start_time = time.time()

        for i in range(num_operations):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"checkpoint-{i}",
            }}
            checkpoint = sample_checkpoint.copy()
            checkpoint["id"] = f"checkpoint-{i}"
            checkpoint["channel_values"]["counter"] = i

            checkpointer.put(config, checkpoint, {"index": i}, {})

        elapsed = time.time() - start_time
        throughput = num_operations / elapsed

        assert throughput > 100  # Expect at least 100 ops/sec
    def test_get_throughput(self, checkpointer) -> None:
        """Test how many gets per second we can handle."""
        # Setup - create checkpoints first
        num_checkpoints = 100
        thread_id = str(uuid.uuid4())

        # Create checkpoints
        for i in range(num_checkpoints):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"checkpoint-{i}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"checkpoint-{i}",
                "channel_values": {"counter": i, "data": f"data-{i}" * 10},
                "channel_versions": {"counter": i + 1},
                "versions_seen": {},
            }
            checkpointer.put(config, checkpoint, {"index": i}, {})

        # Measure get throughput
        num_operations = 1000
        start_time = time.time()

        for i in range(num_operations):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"checkpoint-{i % num_checkpoints}",
            }}
            result = checkpointer.get_tuple(config)
            assert result is not None

        elapsed = time.time() - start_time
        throughput = num_operations / elapsed

        assert throughput > 500  # Expect at least 500 ops/sec for reads
    def test_list_throughput(self, checkpointer) -> None:
        """Test list operation throughput with varying result sizes."""
        thread_id = str(uuid.uuid4())

        # Test with different checkpoint counts
        checkpoint_counts = [10, 50, 100]

        for count in checkpoint_counts:
            # Create checkpoints
            for i in range(count):
                config = {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": f"checkpoint-{i}",
                }}
                checkpoint = {
                    "v": 4,
                    "ts": datetime.utcnow().isoformat(),
                    "id": f"checkpoint-{i}",
                    "channel_values": {"index": i},
                    "channel_versions": {"index": i + 1},
                    "versions_seen": {},
                }
                checkpointer.put(config, checkpoint, {"step": i}, {})

            # Measure list throughput
            num_operations = 100
            start_time = time.time()

            for _ in range(num_operations):
                config = {"configurable": {"thread_id": thread_id}}
                results = list(checkpointer.list(config))
                assert len(results) == count

            elapsed = time.time() - start_time
            num_operations / elapsed


            # Clear for next test
            # Note: In real implementation, we'd need a clear method

class TestScalability:
    """Test how performance scales with data size."""

    def test_scalability_with_checkpoint_count(self, checkpointer) -> None:
        """Test how performance degrades with increasing checkpoint count."""
        thread_id = str(uuid.uuid4())
        checkpoint_counts = [10, 100, 1000, 5000]

        put_times = []
        get_times = []
        list_times = []

        for count in checkpoint_counts:
            # Measure put time for batch
            start_time = time.time()

            for i in range(count):
                config = {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": f"checkpoint-{i}",
                }}
                checkpoint = {
                    "v": 4,
                    "ts": datetime.utcnow().isoformat(),
                    "id": f"checkpoint-{i}",
                    "channel_values": {
                        "counter": i,
                        "message": f"Message {i}",
                        "metadata": {"index": i, "batch": count},
                    },
                    "channel_versions": {"counter": i + 1},
                    "versions_seen": {},
                }
                checkpointer.put(config, checkpoint, {"batch_size": count}, {})

            put_time = time.time() - start_time
            put_times.append(put_time)
            # Measure get time
            start_time = time.time()
            # Sample 100 random gets
            for _ in range(100):
                idx = uuid.uuid4().int % count
                config = {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": f"checkpoint-{idx}",
                }}
                result = checkpointer.get_tuple(config)
                assert result is not None

            get_time = (time.time() - start_time) / 100  # Average per operation
            get_times.append(get_time)

            # Measure list time
            start_time = time.time()
            config = {"configurable": {"thread_id": thread_id}}
            results = list(checkpointer.list(config))
            assert len(results) == count

            list_time = time.time() - start_time
            list_times.append(list_time)

        # Print results

        for i, count in enumerate(checkpoint_counts):
            pass

        # Check that performance doesn't degrade too badly
        # Get time should remain relatively constant
        assert max(get_times) < min(get_times) * 3  # No more than 3x degradation

        # List time should scale linearly or better
        # Check that doubling checkpoints doesn't more than triple list time
        if len(list_times) > 1:
            for i in range(1, len(list_times)):
                ratio = list_times[i] / list_times[i-1]
                size_ratio = checkpoint_counts[i] / checkpoint_counts[i-1]
                assert ratio < size_ratio * 1.5  # Allow 50% overhead

class TestConcurrency:
    """Test concurrent access patterns."""

    def test_concurrent_puts(self, checkpointer) -> None:
        """Test multiple threads putting checkpoints simultaneously."""
        num_threads = 10
        operations_per_thread = 100
        thread_id = str(uuid.uuid4())

        def worker(worker_id: int) -> None:
            """Worker function for concurrent puts."""
            for i in range(operations_per_thread):
                config = {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": f"worker-{worker_id}-checkpoint-{i}",
                }}
                checkpoint = {
                    "v": 4,
                    "ts": datetime.utcnow().isoformat(),
                    "id": f"worker-{worker_id}-checkpoint-{i}",
                    "channel_values": {
                        "worker": worker_id,
                        "operation": i,
                    },
                    "channel_versions": {"operation": i + 1},
                    "versions_seen": {},
                }
                checkpointer.put(config, checkpoint, {"worker": worker_id}, {})

        # Run concurrent puts
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        elapsed = time.time() - start_time
        total_operations = num_threads * operations_per_thread
        total_operations / elapsed


        # Verify all checkpoints were stored
        config = {"configurable": {"thread_id": thread_id}}
        results = list(checkpointer.list(config))
        assert len(results) == total_operations

    def test_concurrent_reads(self, checkpointer) -> None:
        """Test multiple threads reading checkpoints simultaneously."""
        # Setup - create checkpoints first
        thread_id = str(uuid.uuid4())
        num_checkpoints = 100

        for i in range(num_checkpoints):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"checkpoint-{i}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"checkpoint-{i}",
                "channel_values": {
                    "index": i,
                    "data": f"Test data {i}" * 10,
                },
                "channel_versions": {"index": i + 1},
                "versions_seen": {},
            }
            checkpointer.put(config, checkpoint, {"setup": True}, {})

        # Test concurrent reads
        num_threads = 20
        reads_per_thread = 100

        def reader(worker_id: int) -> None:
            """Worker function for concurrent reads."""
            for _ in range(reads_per_thread):
                # Read random checkpoint
                idx = uuid.uuid4().int % num_checkpoints
                config = {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": f"checkpoint-{idx}",
                }}
                result = checkpointer.get_tuple(config)
                assert result is not None
                assert result.checkpoint["channel_values"]["index"] == idx

        # Run concurrent reads
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reader, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        elapsed = time.time() - start_time
        total_operations = num_threads * reads_per_thread
        total_operations / elapsed


    def test_mixed_operations(self, checkpointer) -> None:
        """Test concurrent mix of reads and writes."""
        thread_id = str(uuid.uuid4())
        num_threads = 10
        operations_per_thread = 50

        # Pre-populate some checkpoints
        for i in range(50):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"initial-{i}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"initial-{i}",
                "channel_values": {"value": i},
                "channel_versions": {"value": 1},
                "versions_seen": {},
            }
            checkpointer.put(config, checkpoint, {}, {})

        def mixed_worker(worker_id: int) -> None:
            """Worker performing mixed operations."""
            for i in range(operations_per_thread):
                if i % 3 == 0:  # Write operation (33%)
                    config = {"configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": f"worker-{worker_id}-new-{i}",
                    }}
                    checkpoint = {
                        "v": 4,
                        "ts": datetime.utcnow().isoformat(),
                        "id": f"worker-{worker_id}-new-{i}",
                        "channel_values": {"worker": worker_id, "op": i},
                        "channel_versions": {"op": i + 1},
                        "versions_seen": {},
                    }
                    checkpointer.put(config, checkpoint, {"type": "write"}, {})
                else:  # Read operation (67%)
                    idx = uuid.uuid4().int % 50
                    config = {"configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": f"initial-{idx}",
                    }}
                    result = checkpointer.get_tuple(config)
                    assert result is not None

        # Run mixed operations
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_worker, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        elapsed = time.time() - start_time
        total_operations = num_threads * operations_per_thread
        total_operations / elapsed



class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_memory_usage_growth(self, checkpointer) -> None:
        """Test memory usage as checkpoints accumulate."""
        import tracemalloc

        thread_id = str(uuid.uuid4())
        checkpoint_counts = [100, 500, 1000, 2000]
        memory_usage = []

        for count in checkpoint_counts:
            # Start memory tracking
            tracemalloc.start()

            # Create checkpoints
            for i in range(count):
                config = {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": f"mem-test-{i}",
                }}
                checkpoint = {
                    "v": 4,
                    "ts": datetime.utcnow().isoformat(),
                    "id": f"mem-test-{i}",
                    "channel_values": {
                        "index": i,
                        "data": "x" * 1000,  # 1KB of data per checkpoint
                        "metadata": {"batch": count},
                    },
                    "channel_versions": {"index": i + 1},
                    "versions_seen": {},
                }
                checkpointer.put(config, checkpoint, {"memory_test": True}, {})

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            memory_usage.append((count, current / 1024 / 1024, peak / 1024 / 1024))  # Convert to MB
            tracemalloc.stop()

        # Print results

        for count, current, peak in memory_usage:
            pass

        # Check that memory usage scales reasonably
        # Memory per checkpoint should be relatively constant
        if len(memory_usage) > 1:
            bytes_per_checkpoint = []
            for i in range(1, len(memory_usage)):
                memory_diff = (memory_usage[i][1] - memory_usage[i-1][1]) * 1024 * 1024
                count_diff = checkpoint_counts[i] - checkpoint_counts[i-1]
                bytes_per_checkpoint.append(memory_diff / count_diff)

            sum(bytes_per_checkpoint) / len(bytes_per_checkpoint)

    def test_memory_cleanup(self, checkpointer) -> None:
        """Test memory cleanup and garbage collection."""
        import gc
        import tracemalloc

        thread_id = str(uuid.uuid4())

        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Create many checkpoints
        for i in range(1000):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"cleanup-test-{i}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"cleanup-test-{i}",
                "channel_values": {
                    "large_data": "x" * 10000,  # 10KB per checkpoint
                    "index": i,
                },
                "channel_versions": {"index": i + 1},
                "versions_seen": {},
            }
            checkpointer.put(config, checkpoint, {}, {})

        peak_memory = tracemalloc.get_traced_memory()[0]

        # Clear references and run garbage collection
        # In a real implementation, we might have a clear() method
        del config
        del checkpoint
        gc.collect()

        # Check memory after cleanup
        after_gc_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Convert to MB for readability
        initial_memory / 1024 / 1024
        peak_mb = peak_memory / 1024 / 1024
        after_gc_mb = after_gc_memory / 1024 / 1024


        # Some memory should be recoverable
        assert after_gc_mb < peak_mb


class TestAsyncPerformance:
    """Test async operation performance."""

    @pytest.mark.asyncio
    async def test_async_put_throughput(self, checkpointer) -> None:
        """Test throughput of async put operations."""
        num_operations = 1000
        thread_id = str(uuid.uuid4())

        start_time = time.time()

        for i in range(num_operations):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"async-checkpoint-{i}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"async-checkpoint-{i}",
                "channel_values": {"counter": i},
                "channel_versions": {"counter": i + 1},
                "versions_seen": {},
            }

            await checkpointer.aput(config, checkpoint, {"async": True}, {})

        elapsed = time.time() - start_time
        throughput = num_operations / elapsed

        assert throughput > 150  # Expect better than sync

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self, checkpointer) -> None:
        """Test concurrent async operations with asyncio.gather()."""
        thread_id = str(uuid.uuid4())
        num_concurrent = 100

        async def async_operation(op_id: int):
            """Single async operation."""
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"concurrent-async-{op_id}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"concurrent-async-{op_id}",
                "channel_values": {"operation": op_id},
                "channel_versions": {"operation": 1},
                "versions_seen": {},
            }

            # Do a put and then a get
            await checkpointer.aput(config, checkpoint, {}, {})
            result = await checkpointer.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == f"concurrent-async-{op_id}"
            return op_id

        # Run many operations concurrently
        start_time = time.time()

        results = await asyncio.gather(
            *[async_operation(i) for i in range(num_concurrent)],
        )

        elapsed = time.time() - start_time
        (num_concurrent * 2) / elapsed  # 2 ops per call


        assert len(results) == num_concurrent
        assert all(results[i] == i for i in range(num_concurrent))

    @pytest.mark.asyncio
    async def test_async_streaming(self, checkpointer) -> None:
        """Test streaming performance with async list operations."""
        thread_id = str(uuid.uuid4())

        # Create many checkpoints
        for i in range(500):
            config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"stream-{i}",
            }}
            checkpoint = {
                "v": 4,
                "ts": datetime.utcnow().isoformat(),
                "id": f"stream-{i}",
                "channel_values": {"index": i},
                "channel_versions": {"index": i + 1},
                "versions_seen": {},
            }
            await checkpointer.aput(config, checkpoint, {"batch": i // 100}, {})

        # Stream through results
        start_time = time.time()
        count = 0

        config = {"configurable": {"thread_id": thread_id}}
        async for _checkpoint_tuple in checkpointer.alist(config):
            count += 1
            # Simulate some processing
            await asyncio.sleep(0.001)

        elapsed = time.time() - start_time
        count / elapsed


        assert count == 500
