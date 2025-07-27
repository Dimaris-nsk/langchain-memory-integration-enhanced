"""Parquet exporter for checkpoint data."""

import asyncio
import json
from datetime import datetime
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base_exporter import BaseExporter, ExportConfig, ExportStats


class ParquetExporter(BaseExporter):
    """Export checkpoints to Parquet format for efficient columnar storage."""

    def __init__(
        self,
        client,
        config: ExportConfig | None = None,
        compression: str = "snappy",
        partitioning: list[str] | None = None,
    ) -> None:
        """
        Initialize Parquet exporter.

        Args:
            client: UnifiedMemoryClient instance
            config: Export configuration
            compression: Compression type ('snappy', 'gzip', 'brotli', 'lz4', 'zstd')
            partitioning: List of columns to partition by (e.g., ['thread_id'])
        """
        super().__init__(client, config)
        self.compression = compression
        self.partitioning = partitioning or []

    def export_checkpoints(self) -> ExportStats:
        """Export checkpoints to Parquet file(s)."""
        start_time = datetime.now()

        try:
            # Ensure output directory exists
            self.config.output_path.mkdir(parents=True, exist_ok=True)

            # Fetch all checkpoints
            checkpoints = self._fetch_all_checkpoints()
            self.stats.total_checkpoints = len(checkpoints)

            # Apply filters
            filtered_checkpoints = self._filter_checkpoints(checkpoints)

            # Convert to DataFrame for Parquet
            df = self._checkpoints_to_dataframe(filtered_checkpoints)

            if df.empty:
                return self.stats

            # Export based on partitioning
            if self.partitioning:
                self._export_partitioned(df)
            else:
                self._export_single_file(df)

        except Exception as e:
            self.stats.errors.append(f"Export failed: {e!s}")
            raise

        finally:
            # Calculate duration
            self.stats.export_duration_seconds = (
                datetime.now() - start_time
            ).total_seconds()

        return self.stats

    def _fetch_all_checkpoints(self) -> list[dict[str, Any]]:
        """Fetch all checkpoint documents from unified-memory."""
        # Create event loop for sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Search for all checkpoint documents
            results = loop.run_until_complete(
                self.client.client.qdrant_client.scroll(
                    collection_name=self.client.collection_name,
                    scroll_filter={
                        "must": [{"key": "type", "match": {"value": "checkpoint"}}],
                    },
                    limit=10000,  # Large limit for getting all
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            checkpoints = []
            for point in results[0]:  # First element is the list of points
                checkpoint_data = point.payload
                checkpoint_data["id"] = point.id
                checkpoints.append(checkpoint_data)

            return checkpoints

        finally:
            loop.close()

    def _checkpoints_to_dataframe(
        self, checkpoints: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert checkpoints to pandas DataFrame with proper schema."""
        if not checkpoints:
            # Return empty DataFrame with schema
            return pd.DataFrame(
                columns=[
                    "id",
                    "thread_id",
                    "checkpoint_id",
                    "parent_checkpoint_id",
                    "type",
                    "timestamp",
                    "checkpoint_data",
                    "metadata",
                ],
            )

        # Extract data for DataFrame
        rows = []
        for checkpoint in checkpoints:
            try:
                # Serialize checkpoint data as binary
                checkpoint_bytes = json.dumps(checkpoint.get("checkpoint", {})).encode(
                    "utf-8",
                )

                # Serialize metadata as JSON string
                metadata_str = json.dumps(checkpoint.get("metadata", {}), default=str)

                # Extract timestamp
                ts = checkpoint.get("ts")
                if isinstance(ts, str):
                    timestamp = pd.to_datetime(ts)
                elif isinstance(ts, (int, float)):
                    timestamp = pd.to_datetime(ts, unit="s")
                else:
                    timestamp = pd.to_datetime("now")

                row = {
                    "id": checkpoint.get("id", ""),
                    "thread_id": checkpoint.get("thread_id", ""),
                    "checkpoint_id": checkpoint.get("checkpoint_id", ""),
                    "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
                    "type": checkpoint.get("type", "checkpoint"),
                    "timestamp": timestamp,
                    "checkpoint_data": checkpoint_bytes,
                    "metadata": metadata_str,
                }
                rows.append(row)
                self.stats.total_bytes += len(checkpoint_bytes) + len(metadata_str)

            except Exception as e:
                self.stats.errors.append(
                    f"Failed to process checkpoint {checkpoint.get('id')}: {e!s}",
                )
                self.stats.skipped_checkpoints += 1

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Ensure proper data types
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _export_single_file(self, df: pd.DataFrame) -> None:
        """Export DataFrame to a single Parquet file."""
        filename = "checkpoints_export.parquet"
        output_file = self.config.output_path / filename

        # Create PyArrow Table with schema
        table = self._create_pyarrow_table(df)

        # Write to Parquet
        pq.write_table(
            table,
            output_file,
            compression=self.compression,
            metadata=self._create_parquet_metadata(),
        )

        self.stats.exported_checkpoints = len(df)

        # Report final progress
        self._report_progress(len(df), len(df))

    def _export_partitioned(self, df: pd.DataFrame) -> None:
        """Export DataFrame as partitioned Parquet dataset."""
        output_dir = self.config.output_path / "partitioned_checkpoints"

        # Create PyArrow Table
        table = self._create_pyarrow_table(df)

        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(output_dir),
            partition_cols=self.partitioning,
            compression=self.compression,
        )

        self.stats.exported_checkpoints = len(df)

        # Report final progress
        self._report_progress(len(df), len(df))

    def _create_pyarrow_table(self, df: pd.DataFrame) -> pa.Table:
        """Create PyArrow Table with proper schema."""
        # Define schema explicitly for better control
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("thread_id", pa.string()),
                ("checkpoint_id", pa.string()),
                ("parent_checkpoint_id", pa.string()),
                ("type", pa.string()),
                ("timestamp", pa.timestamp("ms")),
                ("checkpoint_data", pa.binary()),
                ("metadata", pa.string()),
            ],
        )

        # Create table from DataFrame with schema
        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


    def _create_parquet_metadata(self) -> dict[str, str]:
        """Create metadata for Parquet file."""
        metadata = {
            "created_by": "UnifiedCheckpointer",
            "export_timestamp": datetime.now().isoformat(),
            "exporter_version": "1.0.0",
            "compression": self.compression,
            "total_checkpoints": str(self.stats.exported_checkpoints),
        }

        if self.config.thread_ids:
            metadata["filtered_threads"] = ",".join(self.config.thread_ids)

        return metadata

    def export_by_thread(self, thread_id: str) -> ExportStats:
        """
        Export checkpoints for a specific thread to Parquet.

        Args:
            thread_id: Thread ID to export

        Returns:
            Export statistics
        """
        # Update config to filter by thread
        self.config.thread_ids = {thread_id}

        # Use thread ID in filename
        original_path = self.config.output_path
        self.config.output_path = original_path / f"thread_{thread_id}"

        try:
            return self.export_checkpoints()
        finally:
            # Restore original path
            self.config.output_path = original_path

    def export_to_batches(self, max_rows_per_file: int = 100000) -> ExportStats:
        """
        Export checkpoints to multiple Parquet files based on row count.

        Args:
            max_rows_per_file: Maximum rows per Parquet file

        Returns:
            Export statistics
        """
        start_time = datetime.now()

        try:
            # Fetch and filter checkpoints
            checkpoints = self._fetch_all_checkpoints()
            self.stats.total_checkpoints = len(checkpoints)
            filtered_checkpoints = self._filter_checkpoints(checkpoints)

            # Process in batches
            batch_num = 0
            for i in range(0, len(filtered_checkpoints), max_rows_per_file):
                batch = filtered_checkpoints[i : i + max_rows_per_file]
                batch_num += 1

                # Convert batch to DataFrame
                df = self._checkpoints_to_dataframe(batch)

                if not df.empty:
                    # Export batch
                    filename = f"checkpoints_batch_{batch_num:04d}.parquet"
                    output_file = self.config.output_path / filename

                    table = self._create_pyarrow_table(df)
                    pq.write_table(table, output_file, compression=self.compression)

                    self.stats.exported_checkpoints += len(df)

                # Report progress
                self._report_progress(
                    min(i + max_rows_per_file, len(filtered_checkpoints)),
                    len(filtered_checkpoints),
                )


        except Exception as e:
            self.stats.errors.append(f"Batch export failed: {e!s}")
            raise

        finally:
            self.stats.export_duration_seconds = (
                datetime.now() - start_time
            ).total_seconds()

        return self.stats
