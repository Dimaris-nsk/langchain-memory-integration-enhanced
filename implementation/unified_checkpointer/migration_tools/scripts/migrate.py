#!/usr/bin/env python3
"""Main migration script for checkpoint data."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from migration_tools.exporters import ExportConfig, JSONExporter, ParquetExporter
from migration_tools.importers import PostgreSQLImporter, RedisImporter, SQLiteImporter
from unified_checkpointer.client import UnifiedMemoryClient
from unified_checkpointer.config import UnifiedCheckpointerConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration for migration operation."""

    source_type: str
    source_config: dict[str, Any]
    destination_type: str
    destination_config: dict[str, Any]

    # Optional settings
    batch_size: int = 1000
    dry_run: bool = False
    filters: dict[str, Any] | None = None
    transformers: list[str] | None = None
    progress_interval: int = 100

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MigrationConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class MigrationManager:
    """Manages the migration process from source to destination."""

    def __init__(self, config: MigrationConfig) -> None:
        """Initialize migration manager with configuration."""
        self.config = config
        self.stats = {
            "total_records": 0,
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }
        self.source = None
        self.destination = None

    def setup_source(self) -> None:
        """Initialize source adapter based on configuration."""
        source_type = self.config.source_type.lower()

        if source_type == "postgresql":
            self.source = PostgreSQLImporter(
                connection_string=self.config.source_config["connection_string"],
            )
        elif source_type == "sqlite":
            self.source = SQLiteImporter(db_path=self.config.source_config["db_path"])
        elif source_type == "redis":
            self.source = RedisImporter(**self.config.source_config)
        elif source_type == "unified":
            # Source is UnifiedCheckpointer
            config = UnifiedCheckpointerConfig(
                qdrant_url=self.config.source_config.get(
                    "url", "http://localhost:6333",
                ),
                collection_name=self.config.source_config.get(
                    "collection", "checkpoints",
                ),
            )
            self.source = UnifiedMemoryClient(config)
        else:
            msg = f"Unknown source type: {source_type}"
            raise ValueError(msg)

    def setup_destination(self) -> None:
        """Initialize destination adapter based on configuration."""
        dest_type = self.config.destination_type.lower()

        if dest_type == "unified":
            # Destination is UnifiedCheckpointer
            config = UnifiedCheckpointerConfig(
                qdrant_url=self.config.destination_config.get(
                    "url", "http://localhost:6333",
                ),
                collection_name=self.config.destination_config.get(
                    "collection", "checkpoints",
                ),
            )
            self.destination = UnifiedMemoryClient(config)

        elif dest_type == "json":
            # JSON export
            export_config = ExportConfig(
                output_path=Path(self.config.destination_config["output_path"]),
                batch_size=self.config.batch_size,
            )
            client = self._get_unified_client_for_export()
            self.destination = JSONExporter(
                client=client,
                config=export_config,
                **self.config.destination_config.get("options", {}),
            )

        elif dest_type == "parquet":
            # Parquet export
            export_config = ExportConfig(
                output_path=Path(self.config.destination_config["output_path"]),
                batch_size=self.config.batch_size,
            )
            client = self._get_unified_client_for_export()
            self.destination = ParquetExporter(
                client=client,
                config=export_config,
                **self.config.destination_config.get("options", {}),
            )
        else:
            msg = f"Unknown destination type: {dest_type}"
            raise ValueError(msg)

    def _get_unified_client_for_export(self) -> UnifiedMemoryClient:
        """Get UnifiedMemoryClient for export operations."""
        # For exports, we need a unified client as source
        if isinstance(self.source, UnifiedMemoryClient):
            return self.source
        # Create temporary unified client for intermediate storage
        config = UnifiedCheckpointerConfig(
            qdrant_url="http://localhost:6333", collection_name="temp_migration",
        )
        return UnifiedMemoryClient(config)

    def run(self):
        """Execute the migration process."""
        self.stats["start_time"] = datetime.now()
        logger.info(
            f"Starting migration: {self.config.source_type} -> {self.config.destination_type}",
        )

        try:
            # Setup source and destination
            self.setup_source()
            self.setup_destination()

            if self.config.dry_run:
                logger.info("DRY RUN MODE - No actual data will be migrated")
                return self._dry_run()

            # Perform actual migration
            self._migrate()

        except Exception as e:
            logger.exception(f"Migration failed: {e!s}")
            raise
        finally:
            self.stats["end_time"] = datetime.now()
            self._print_summary()

    def _migrate(self) -> None:
        """Perform the actual migration."""
        logger.info("Starting data migration...")

        try:
            # Get total count for progress tracking
            total_count = self._get_source_count()
            self.stats["total_records"] = total_count
            logger.info(f"Found {total_count} records to migrate")

            # Initialize progress counter
            processed = 0
            batch = []

            # Iterate through source checkpoints
            for checkpoint_data in self._iterate_source():
                try:
                    # Apply filters if configured
                    if self.config.filters and not self._apply_filters(checkpoint_data):
                        self.stats["skipped"] += 1
                        continue

                    # Apply transformers if configured
                    if self.config.transformers:
                        checkpoint_data = self._apply_transformers(checkpoint_data)

                    # Add to batch
                    batch.append(checkpoint_data)

                    # Process batch when full
                    if len(batch) >= self.config.batch_size:
                        self._process_batch(batch)
                        processed += len(batch)
                        batch = []

                        # Progress update
                        if processed % self.config.progress_interval == 0:
                            pct = (
                                (processed / total_count) * 100
                                if total_count > 0
                                else 0
                            )
                            logger.info(
                                f"Progress: {processed}/{total_count} ({pct:.1f}%)",
                            )

                except Exception as e:
                    logger.exception(f"Error processing checkpoint: {e!s}")
                    self.stats["errors"] += 1
                    if self.stats["errors"] > 100:  # Fail if too many errors
                        msg = "Too many errors during migration"
                        raise RuntimeError(msg)

            # Process remaining batch
            if batch:
                self._process_batch(batch)
                processed += len(batch)

            logger.info(f"Migration complete: {processed} records processed")

        except Exception as e:
            logger.exception(f"Migration failed: {e!s}")
            raise

    def _dry_run(self) -> None:
        """Perform a dry run without actual migration."""
        logger.info("Performing dry run analysis...")

        try:
            # Count total records
            total_count = self._get_source_count()
            self.stats["total_records"] = total_count
            logger.info(f"Source contains {total_count} records")

            # Sample analysis
            sample_size = min(100, total_count)
            samples = []

            for i, checkpoint_data in enumerate(self._iterate_source()):
                if i >= sample_size:
                    break

                # Check filters
                if self.config.filters and not self._apply_filters(checkpoint_data):
                    self.stats["skipped"] += 1
                else:
                    self.stats["migrated"] += 1
                    samples.append(checkpoint_data)

            # Estimate final counts
            if sample_size > 0:
                skip_rate = self.stats["skipped"] / sample_size
                self.stats["skipped"] = int(total_count * skip_rate)
                self.stats["migrated"] = total_count - self.stats["skipped"]

            # Report findings
            logger.info("\nDry Run Results:")
            logger.info(f"- Total records: {total_count}")
            logger.info(f"- Would migrate: {self.stats['migrated']}")
            logger.info(f"- Would skip: {self.stats['skipped']}")

            # Check destination
            if hasattr(self.destination, "validate_connection"):
                if self.destination.validate_connection():
                    logger.info("- Destination: ✓ Connection validated")
                else:
                    logger.warning("- Destination: ✗ Connection failed")

        except Exception as e:
            logger.exception(f"Dry run failed: {e!s}")
            raise

    def _print_summary(self) -> None:
        """Print migration summary statistics."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        logger.info("\n" + "=" * 50)
        logger.info("Migration Summary:")
        logger.info(f"Total records: {self.stats['total_records']}")
        logger.info(f"Migrated: {self.stats['migrated']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 50)

    def _get_source_count(self) -> int:
        """Get total count of records from source."""
        if hasattr(self.source, "get_count"):
            return self.source.get_count()
        if hasattr(self.source, "count_checkpoints"):
            return self.source.count_checkpoints()
        # Count by iterating (slower)
        count = 0
        for _ in self._iterate_source():
            count += 1
        return count

    def _iterate_source(self):
        """Iterate through source checkpoints."""
        source_type = self.config.source_type.lower()

        if source_type in ["postgresql", "sqlite", "redis"]:
            # Use importer's import_conversations method
            for checkpoint in self.source.import_conversations():
                yield checkpoint

        elif source_type == "unified":
            # Read from UnifiedCheckpointer
            limit = 1000
            offset = 0
            while True:
                checkpoints = self.source.list_checkpoints(limit=limit, offset=offset)
                if not checkpoints:
                    break
                for checkpoint in checkpoints:
                    yield checkpoint
                offset += limit
        else:
            msg = f"Unknown source type: {source_type}"
            raise ValueError(msg)

    def _apply_filters(self, checkpoint_data: dict[str, Any]) -> bool:
        """Apply filters to checkpoint data."""
        if not self.config.filters:
            return True

        # Example filters
        if "thread_id" in self.config.filters:
            if checkpoint_data.get("thread_id") != self.config.filters["thread_id"]:
                return False

        if "date_from" in self.config.filters:
            checkpoint_ts = checkpoint_data.get("timestamp", "")
            if checkpoint_ts < self.config.filters["date_from"]:
                return False

        return True

    def _apply_transformers(self, checkpoint_data: dict[str, Any]) -> dict[str, Any]:
        """Apply transformers to checkpoint data."""
        if not self.config.transformers:
            return checkpoint_data

        # Apply each transformer in sequence
        for transformer_name in self.config.transformers:
            if transformer_name == "add_metadata":
                checkpoint_data["metadata"] = checkpoint_data.get("metadata", {})
                checkpoint_data["metadata"]["migrated_at"] = datetime.now().isoformat()
            # Add more transformers as needed

        return checkpoint_data

    def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        """Process a batch of checkpoints."""
        dest_type = self.config.destination_type.lower()

        try:
            if dest_type == "unified":
                # Write to UnifiedCheckpointer
                for checkpoint in batch:
                    self.destination.store_checkpoint(
                        thread_id=checkpoint["thread_id"],
                        checkpoint_id=checkpoint.get("checkpoint_id"),
                        checkpoint=checkpoint["checkpoint"],
                        metadata=checkpoint.get("metadata", {}),
                    )

            elif dest_type in ["json", "parquet"]:
                # Use exporter's export method
                self.destination.export_checkpoints(batch)

            self.stats["migrated"] += len(batch)

        except Exception as e:
            logger.exception(f"Failed to process batch: {e!s}")
            self.stats["errors"] += len(batch)
            raise


def main() -> None:
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate checkpoint data between different storage systems",
    )

    # Source and destination arguments
    parser.add_argument(
        "--source",
        required=True,
        help="Source type (postgresql, sqlite, redis, unified)",
    )
    parser.add_argument(
        "--source-config",
        type=json.loads,
        required=True,
        help="Source configuration as JSON",
    )
    parser.add_argument(
        "--destination", required=True, help="Destination type (unified, json, parquet)",
    )
    parser.add_argument(
        "--dest-config",
        type=json.loads,
        required=True,
        help="Destination configuration as JSON",
    )

    # Optional arguments
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual migration",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for migration",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    if args.config:
        config = MigrationConfig.from_yaml(args.config)
    else:
        config = MigrationConfig(
            source_type=args.source,
            source_config=args.source_config,
            destination_type=args.destination,
            destination_config=args.dest_config,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

    # Run migration
    manager = MigrationManager(config)
    manager.run()


if __name__ == "__main__":
    main()