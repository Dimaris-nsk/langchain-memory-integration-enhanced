#!/usr/bin/env python3
"""
Example migration scripts using the migrate.py tool.

This file demonstrates various migration scenarios and patterns.
"""

import json
import subprocess


def run_migration(
    source: str,
    source_config: dict,
    destination: str,
    dest_config: dict,
    dry_run: bool = True,
):
    """Helper function to run migration with proper arguments."""
    cmd = [
        "python",
        "-m",
        "unified_checkpointer.migration_tools.scripts.migrate",
        "--source",
        source,
        "--source-config",
        json.dumps(source_config),
        "--destination",
        destination,
        "--dest-config",
        json.dumps(dest_config),
    ]

    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        pass
    else:
        pass

    return result.returncode == 0


# Example 1: PostgreSQL to Unified Memory
def example_postgresql_to_unified() -> None:
    """Migrate from PostgreSQL to Unified Memory (Qdrant)."""

    source_config = {
        "connection_string": "postgresql://user:pass@localhost/checkpoints",
        "table_name": "chat_message_history",
    }

    dest_config = {
        "qdrant_url": "http://localhost:6333",
        "collection": "checkpoints",
        "embedding_model": "text-embedding-ada-002",
    }

    # First do a dry run
    run_migration("postgresql", source_config, "unified", dest_config, dry_run=True)

    # Then actual migration (commented out for safety)
    # print("\nActual migration:")
    # run_migration('postgresql', source_config, 'unified', dest_config, dry_run=False)


# Example 2: SQLite to JSON export
def example_sqlite_to_json() -> None:
    """Export SQLite checkpoints to JSON for backup/analysis."""

    source_config = {"connection_string": "sqlite:///checkpoints.db"}

    dest_config = {
        "output_dir": "./exports",
        "single_file": False,  # Create one file per thread
        "indent": 2,
    }

    run_migration("sqlite", source_config, "json", dest_config, dry_run=False)


# Example 3: Redis to Parquet for analytics
def example_redis_to_parquet() -> None:
    """Export Redis checkpoints to Parquet for data analytics."""

    source_config = {
        "redis_url": "redis://localhost:6379",
        "key_prefix": "checkpoint:",
        "decode_responses": True,
    }

    dest_config = {
        "output_path": "./analytics/checkpoints.parquet",
        "compression": "snappy",
        "partition_by": ["thread_id"],
    }

    run_migration("redis", source_config, "parquet", dest_config, dry_run=False)

    # Example 4: Unified to Unified (migration between Qdrant instances)def example_unified_to_unified():
    """Migrate between different Qdrant instances."""

    source_config = {
        "qdrant_url": "http://old-server:6333",
        "collection": "old_checkpoints",
    }

    dest_config = {
        "qdrant_url": "http://new-server:6333",
        "collection": "new_checkpoints",
        "embedding_model": "text-embedding-ada-002",
    }

    # With filters - only migrate specific threads
    filters = {
        "thread_id": "customer-support-*",  # Wildcard pattern
        "date_from": "2025-01-01",
    }

    cmd = [
        "python",
        "-m",
        "unified_checkpointer.migration_tools.scripts.migrate",
        "--source",
        "unified",
        "--source-config",
        json.dumps(source_config),
        "--destination",
        "unified",
        "--dest-config",
        json.dumps(dest_config),
        "--filters",
        json.dumps(filters),
        "--batch-size",
        "500",
    ]

    subprocess.run(cmd, check=False)


# Example 5: Using YAML configuration
def example_yaml_config() -> None:
    """Use YAML configuration file for complex migrations."""

    yaml_config = """
source_type: postgresql
source_config:
  connection_string: postgresql://user:pass@localhost/db
  table_name: chat_history

destination_type: unified
destination_config:
  qdrant_url: http://localhost:6333
  collection: migrated_checkpoints

filters:
  date_from: "2025-01-01"
  exclude_threads:
    - test-*
    - debug-*

transformers:
  - add_metadata
  - normalize_timestamps

batch_size: 1000
progress_interval: 100
"""

    # Save YAML config
    with open("migration_config.yaml", "w") as f:
        f.write(yaml_config)

    # Run with YAML config
    cmd = [
        "python",
        "-m",
        "unified_checkpointer.migration_tools.scripts.migrate",
        "--config",
        "migration_config.yaml",
    ]

    subprocess.run(cmd, check=False)


# Example 6: Programmatic usage (without CLI)
def example_programmatic() -> None:
    """Use MigrationManager directly in Python code."""

    from unified_checkpointer.migration_tools.scripts.migrate import (
        MigrationConfig,
        MigrationManager,
    )

    # Create configuration
    config = MigrationConfig(
        source_type="sqlite",
        source_config={"connection_string": "sqlite:///local.db"},
        destination_type="json",
        destination_config={"output_dir": "./backups", "single_file": True},
        batch_size=500,
        dry_run=False,
    )

    # Run migration
    manager = MigrationManager(config)
    manager.run()



if __name__ == "__main__":
    # Run examples (comment out ones you don't want to run)
    example_postgresql_to_unified()
    example_sqlite_to_json()
    example_redis_to_parquet()
    example_unified_to_unified()
    example_yaml_config()
    example_programmatic()
