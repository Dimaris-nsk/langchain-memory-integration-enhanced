"""Performance benchmarks for Milestone 3 features.

Tests performance improvements from:
- LRU Cache
- Batch Operations
- Connection Pooling
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from unified_checkpointer import UnifiedCheckpointer
from unified_checkpointer.checkpointer import Checkpoint, CheckpointMetadata
from unified_checkpointer.config import UnifiedCheckpointerConfig


class TestCachePerformance:
    """Test performance improvements from LRU cache."""

    def test_cache_hit_performance(self, checkpointer_with_cache, sample_checkpoint) -> None:
        """Test performance improvement from cache hits."""
        thread_id = str(uuid.uuid4())

        # First access - cache miss
        config = {"configurable": {"thread_id": thread_id}}
        updated_config = checkpointer_with_cache.put(
            config, sample_checkpoint, CheckpointMetadata(), {},
        )
        checkpoint_id = updated_config["configurable"]["checkpoint_id"]

        # Measure time for cache miss
        config_with_id = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
        }

        start = time.perf_counter()
        result = checkpointer_with_cache.get_tuple(config_with_id)
        cache_miss_time = time.perf_counter() - start

        # Second access - cache hit
        start = time.perf_counter()
        result2 = checkpointer_with_cache.get_tuple(config_with_id)
        cache_hit_time = time.perf_counter() - start

        # Cache hit should be at least 10x faster
        speedup = cache_miss_time / cache_hit_time
        assert speedup > 10, f"Cache speedup only {speedup:.2f}x"
        assert result.checkpoint.id == result2.checkpoint.id

    def test_cache_effectiveness_under_load(self, checkpointer_with_cache) -> None:
        """Test cache effectiveness with many threads."""
        num_threads = 20
        reads_per_thread = 50

        # Create checkpoints
        thread_ids = [str(uuid.uuid4()) for _ in range(num_threads)]
        checkpoint_ids = {}

        for thread_id in thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = Checkpoint(
                v=1,
                ts=str(time.time()),
                channel_values={"messages": f"Thread {thread_id}"},
            )
            updated_config = checkpointer_with_cache.put(
                config, checkpoint, CheckpointMetadata(), {},
            )
            checkpoint_ids[thread_id] = updated_config["configurable"]["checkpoint_id"]

        # Function to read checkpoints
        def read_checkpoints(thread_id):
            times = []
            for _ in range(reads_per_thread):
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_ids[thread_id],
                    },
                }
                start = time.perf_counter()
                checkpointer_with_cache.get_tuple(config)
                times.append(time.perf_counter() - start)
            return times

        # Run concurrent reads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            all_times = list(executor.map(read_checkpoints, thread_ids))

        # Analyze results
        for thread_times in all_times:
            # First read might be cache miss, rest should be hits
            avg_first = thread_times[0]
            avg_rest = sum(thread_times[1:]) / len(thread_times[1:])
            speedup = avg_first / avg_rest
            assert speedup > 5, f"Cache not effective: {speedup:.2f}x"


class TestBatchPerformance:
    """Test performance improvements from batch operations."""

    def test_batch_vs_individual_puts(self, checkpointer) -> None:
        """Compare batch put vs individual puts."""
        num_checkpoints = 100

        # Prepare data
        configs = []
        checkpoints = []
        metadatas = []
        new_versions_list = []

        for i in range(num_checkpoints):
            thread_id = str(uuid.uuid4())
            configs.append({"configurable": {"thread_id": thread_id}})
            checkpoints.append(Checkpoint(
                v=1,
                ts=str(time.time()),
                channel_values={"messages": f"Checkpoint {i}"},
            ))
            metadatas.append(CheckpointMetadata())
            new_versions_list.append({})

        # Time individual puts
        start = time.perf_counter()
        for i in range(num_checkpoints):
            checkpointer.put(configs[i], checkpoints[i], metadatas[i], new_versions_list[i])
        individual_time = time.perf_counter() - start

        # Time batch put
        start = time.perf_counter()
        checkpointer.put_batch(configs, checkpoints, metadatas, new_versions_list)
        batch_time = time.perf_counter() - start

        # Batch should be at least 5x faster
        speedup = individual_time / batch_time
        assert speedup > 5, f"Batch speedup only {speedup:.2f}x"

    def test_batch_get_performance(self, checkpointer) -> None:
        """Test batch get performance."""
        num_checkpoints = 50

        # Create checkpoints
        configs = []
        for i in range(num_checkpoints):
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = Checkpoint(
                v=1,
                ts=str(time.time()),
                channel_values={"messages": f"Checkpoint {i}"},
            )
            updated_config = checkpointer.put(config, checkpoint, CheckpointMetadata(), {})
            configs.append(updated_config)

        # Time individual gets
        start = time.perf_counter()
        results = []
        for config in configs:
            result = checkpointer.get_tuple(config)
            results.append(result)
        individual_time = time.perf_counter() - start

        # Time batch get
        start = time.perf_counter()
        batch_results = checkpointer.get_batch(configs)
        batch_time = time.perf_counter() - start

        # Verify results match
        assert len(batch_results) == len(results)
        for i, (single, batch) in enumerate(zip(results, batch_results, strict=False)):
            assert single.checkpoint.id == batch.checkpoint.id

        # Batch should be faster
        speedup = individual_time / batch_time
        assert speedup > 3, f"Batch get speedup only {speedup:.2f}x"


class TestConnectionPoolPerformance:
    """Test performance improvements from connection pooling."""

    @pytest.mark.asyncio
    async def test_pool_concurrent_performance(self, checkpointer_with_pool) -> None:
        """Test connection pool under concurrent load."""
        num_concurrent = 50
        operations_per_task = 10

        async def worker(worker_id):
            """Perform multiple operations."""
            times = []
            thread_id = str(uuid.uuid4())

            for i in range(operations_per_task):
                checkpoint = Checkpoint(
                    v=i + 1,
                    ts=str(time.time()),
                    channel_values={"worker": worker_id, "op": i},
                )

                start = time.perf_counter()
                config = {"configurable": {"thread_id": thread_id}}
                await checkpointer_with_pool.aput(
                    config, checkpoint, CheckpointMetadata(), {},
                )
                times.append(time.perf_counter() - start)

            return times

        # Run concurrent workers
        tasks = [worker(i) for i in range(num_concurrent)]
        all_times = await asyncio.gather(*tasks)

        # Analyze performance
        avg_times = [sum(times) / len(times) for times in all_times]
        overall_avg = sum(avg_times) / len(avg_times)

        # With pooling, average time should stay low even under load
        assert overall_avg < 0.1, f"Average operation time too high: {overall_avg:.3f}s"

        # Check consistency - no worker should be significantly slower
        max_avg = max(avg_times)
        min_avg = min(avg_times)
        ratio = max_avg / min_avg
        assert ratio < 3, f"Too much variance between workers: {ratio:.2f}x"

    def test_pool_vs_single_connection(self) -> None:
        """Compare pooled vs non-pooled performance."""
        # Create two checkpointers - one with pool, one without
        config_with_pool = UnifiedCheckpointerConfig(
            unified_memory_url=None,  # Use in-memory mode
            collection_name="test_pool_perf",
            pool_enabled=True,
            pool_max_connections=10,
        )
        config_no_pool = UnifiedCheckpointerConfig(
            unified_memory_url=None,  # Use in-memory mode
            collection_name="test_no_pool_perf",
            pool_enabled=False,
        )

        with UnifiedCheckpointer(config_with_pool) as checkpointer_pool:
            with UnifiedCheckpointer(config_no_pool) as checkpointer_no_pool:
                # Test concurrent operations
                num_threads = 20
                ops_per_thread = 20

                def worker(checkpointer, thread_num):
                    thread_id = str(uuid.uuid4())
                    times = []

                    for i in range(ops_per_thread):
                        checkpoint = Checkpoint(
                            v=i + 1,
                            ts=str(time.time()),
                            channel_values={"thread": thread_num, "op": i},
                        )
                        config = {"configurable": {"thread_id": thread_id}}

                        start = time.perf_counter()
                        checkpointer.put(config, checkpoint, CheckpointMetadata(), {})
                        times.append(time.perf_counter() - start)

                    return times

                # Test with pool
                start_pool = time.perf_counter()
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    pool_results = list(executor.map(
                        lambda i: worker(checkpointer_pool, i),
                        range(num_threads),
                    ))
                pool_time = time.perf_counter() - start_pool

                # Test without pool
                start_no_pool = time.perf_counter()
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    no_pool_results = list(executor.map(
                        lambda i: worker(checkpointer_no_pool, i),
                        range(num_threads),
                    ))
                no_pool_time = time.perf_counter() - start_no_pool

                # Pool should be faster under concurrent load
                speedup = no_pool_time / pool_time
                assert speedup > 1.5, f"Pool speedup only {speedup:.2f}x"

                # Pool should have more consistent performance
                pool_all_times = [t for thread_times in pool_results for t in thread_times]
                no_pool_all_times = [t for thread_times in no_pool_results for t in thread_times]

                pool_variance = max(pool_all_times) / min(pool_all_times)
                no_pool_variance = max(no_pool_all_times) / min(no_pool_all_times)

                assert pool_variance < no_pool_variance, "Pool should have more consistent performance"


# Fixtures
@pytest.fixture
def checkpointer_with_cache():
    """Create checkpointer with cache enabled."""
    config = UnifiedCheckpointerConfig(
        unified_memory_url=None,  # Use in-memory mode
        collection_name="test_cache_perf",
        cache_enabled=True,
        cache_max_size=1000,
        cache_ttl_seconds=300,
    )
    with UnifiedCheckpointer(config) as checkpointer:
        yield checkpointer


@pytest.fixture
def checkpointer_with_pool():
    """Create checkpointer with connection pool."""
    config = UnifiedCheckpointerConfig(
        unified_memory_url=None,  # Use in-memory mode
        collection_name="test_pool",
        pool_enabled=True,
        pool_max_connections=10,
        pool_connection_ttl=30.0,
    )
    with UnifiedCheckpointer(config) as checkpointer:
        yield checkpointer


class TestCombinedPerformance:
    """Test performance with all Milestone 3 features combined."""

    def test_all_features_combined(self) -> None:
        """Test cache + batch + pool working together."""
        config = UnifiedCheckpointerConfig(
            unified_memory_url=None,  # Use in-memory mode
            collection_name="test_combined",
            # Enable all features
            cache_enabled=True,
            cache_max_size=1000,
            pool_enabled=True,
            pool_max_connections=10,
        )
        with UnifiedCheckpointer(config) as checkpointer:
            # Prepare batch data
            num_batches = 5
            batch_size = 20
            all_configs = []
            all_checkpoints = []

            for batch in range(num_batches):
                configs = []
                checkpoints = []
                metadatas = []
                new_versions = []

                for i in range(batch_size):
                    thread_id = f"thread_{batch}_{i}"
                    configs.append({"configurable": {"thread_id": thread_id}})
                    checkpoints.append(Checkpoint(
                        v=1,
                        ts=str(time.time()),
                        channel_values={"batch": batch, "item": i},
                    ))
                    metadatas.append(CheckpointMetadata())
                    new_versions.append({})

                # Time batch put
                start = time.perf_counter()
                updated_configs = checkpointer.put_batch(
                    configs, checkpoints, metadatas, new_versions,
                )
                time.perf_counter() - start

                all_configs.extend(updated_configs)
                all_checkpoints.extend(checkpoints)


            # Test concurrent reads with cache
            def read_worker(configs_slice):
                times = []
                cache_hits = 0

                for _ in range(3):  # Read each checkpoint 3 times
                    for config in configs_slice:
                        start = time.perf_counter()
                        checkpointer.get_tuple(config)
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)

                        # Heuristic: cache hit if < 1ms
                        if elapsed < 0.001:
                            cache_hits += 1

                return times, cache_hits

            # Split configs among workers
            num_workers = 10
            configs_per_worker = len(all_configs) // num_workers

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(num_workers):
                    start_idx = i * configs_per_worker
                    end_idx = start_idx + configs_per_worker
                    configs_slice = all_configs[start_idx:end_idx]
                    futures.append(executor.submit(read_worker, configs_slice))

                results = [f.result() for f in futures]

            # Analyze results
            all_times = []
            total_cache_hits = 0

            for times, cache_hits in results:
                all_times.extend(times)
                total_cache_hits += cache_hits

            # Calculate percentiles
            all_times.sort()
            p50 = all_times[len(all_times) // 2]
            p95 = all_times[int(len(all_times) * 0.95)]
            all_times[int(len(all_times) * 0.99)]


            # Assertions
            assert p50 < 0.005, f"p50 latency too high: {p50*1000:.2f}ms"
            assert p95 < 0.020, f"p95 latency too high: {p95*1000:.2f}ms"
            assert total_cache_hits > len(all_times) * 0.5, "Cache hit rate too low"

    @pytest.mark.asyncio
    async def test_async_all_features(self) -> None:
        """Test all features with async operations."""
        config = UnifiedCheckpointerConfig(
            unified_memory_url=None,  # Use in-memory mode
            collection_name="test_async_combined",
            cache_enabled=True,
            cache_max_size=500,
            pool_enabled=True,
            pool_max_connections=20,
        )
        checkpointer = UnifiedCheckpointer(config)

        # Create many checkpoints concurrently
        async def create_checkpoints(worker_id, count):
            thread_id = f"async_thread_{worker_id}"
            checkpoint_ids = []

            for i in range(count):
                checkpoint = Checkpoint(
                    v=i + 1,
                    ts=str(time.time()),
                    channel_values={"worker": worker_id, "seq": i},
                )
                config = {"configurable": {"thread_id": thread_id}}

                start = time.perf_counter()
                updated_config = await checkpointer.aput(
                    config, checkpoint, CheckpointMetadata(), {},
                )
                time.perf_counter() - start

                checkpoint_ids.append(updated_config["configurable"]["checkpoint_id"])

                if i % 10 == 0:
                    pass

            return checkpoint_ids

        # Create checkpoints with multiple workers
        num_workers = 10
        checkpoints_per_worker = 50

        create_tasks = [
            create_checkpoints(i, checkpoints_per_worker)
            for i in range(num_workers)
        ]

        start_create = time.perf_counter()
        all_checkpoint_ids = await asyncio.gather(*create_tasks)
        create_time = time.perf_counter() - start_create

        total_checkpoints = num_workers * checkpoints_per_worker
        create_rate = total_checkpoints / create_time


        # Read checkpoints with cache warming
        async def read_checkpoints(worker_id, thread_id, checkpoint_ids):
            read_times = []

            for checkpoint_id in checkpoint_ids:
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id,
                    },
                }

                start = time.perf_counter()
                await checkpointer.aget_tuple(config)
                elapsed = time.perf_counter() - start
                read_times.append(elapsed)

            return read_times

        # Read all checkpoints
        read_tasks = []
        for worker_id, checkpoint_ids in enumerate(all_checkpoint_ids):
            thread_id = f"async_thread_{worker_id}"
            read_tasks.append(read_checkpoints(worker_id, thread_id, checkpoint_ids))

        start_read = time.perf_counter()
        all_read_times = await asyncio.gather(*read_tasks)
        read_time = time.perf_counter() - start_read

        # Flatten times and calculate stats
        all_times = []
        for times in all_read_times:
            all_times.extend(times)

        all_times.sort()
        p50 = all_times[len(all_times) // 2]
        all_times[int(len(all_times) * 0.95)]
        all_times[int(len(all_times) * 0.99)]

        read_rate = total_checkpoints / read_time


        # Performance assertions
        assert create_rate > 100, f"Create rate too low: {create_rate:.0f}/sec"
        assert read_rate > 500, f"Read rate too low: {read_rate:.0f}/sec"
        assert p50 < 0.010, f"p50 read latency too high: {p50*1000:.2f}ms"


def calculate_percentiles(times: list[float], percentiles: list[int] | None = None):
    """Calculate percentile values from a list of times."""
    if percentiles is None:
        percentiles = [50, 95, 99]
    if not times:
        return {}

    sorted_times = sorted(times)
    results = {}

    for p in percentiles:
        idx = int(len(sorted_times) * p / 100)
        if idx >= len(sorted_times):
            idx = len(sorted_times) - 1
        results[f"p{p}"] = sorted_times[idx]

    return results


def print_performance_summary(test_name: str, metrics: dict) -> None:
    """Print a formatted performance summary."""

    for value in metrics.values():
        if isinstance(value, float):
            if value < 0.001 or value < 1:
                pass
            else:
                pass
        else:
            pass



if __name__ == "__main__":
    # Run specific test class or all tests
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main(["-v", "-s", __file__, f"-k {test_class}"])
    else:
        pytest.main(["-v", "-s", __file__])
