"""JSON exporter for checkpoint data."""

import asyncio
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .base_exporter import BaseExporter, ExportConfig, ExportStats


class JSONExporter(BaseExporter):
    """Export checkpoints to JSON format."""

    def __init__(
        self,
        client,
        config: ExportConfig | None = None,
        pretty_print: bool = True,
        compress: bool = False,
    ) -> None:
        """
        Initialize JSON exporter.

        Args:
            client: UnifiedMemoryClient instance
            config: Export configuration
            pretty_print: Whether to format JSON for readability
            compress: Whether to gzip compress the output
        """
        super().__init__(client, config)
        self.pretty_print = pretty_print
        self.compress = compress

    def export_checkpoints(self) -> ExportStats:
        """Export checkpoints to JSON file(s)."""
        start_time = datetime.now()

        try:
            # Ensure output directory exists
            self.config.output_path.mkdir(parents=True, exist_ok=True)

            # Fetch all checkpoints
            checkpoints = self._fetch_all_checkpoints()
            self.stats.total_checkpoints = len(checkpoints)

            # Apply filters
            filtered_checkpoints = self._filter_checkpoints(checkpoints)

            # Export based on batch size
            if len(filtered_checkpoints) <= self.config.batch_size:
                # Single file export
                self._export_single_file(filtered_checkpoints)
            else:
                # Multi-file export
                self._export_multi_file(filtered_checkpoints)

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

    def _export_single_file(self, checkpoints: list[dict[str, Any]]) -> None:
        """Export all checkpoints to a single JSON file."""
        filename = "checkpoints_export.json"
        if self.compress:
            filename += ".gz"

        output_file = self.config.output_path / filename
        # Prepare export data
        export_data = {
            "metadata": self._create_export_metadata()
            if self.config.include_metadata
            else {},
            "checkpoints": [],
        }

        # Serialize each checkpoint
        for i, checkpoint in enumerate(checkpoints):
            try:
                serialized = self._serialize_checkpoint(checkpoint)
                export_data["checkpoints"].append(serialized)
                self.stats.exported_checkpoints += 1

                # Report progress
                self._report_progress(i + 1, len(checkpoints))

            except Exception as e:
                self.stats.errors.append(
                    f"Failed to serialize checkpoint {checkpoint.get('id')}: {e!s}",
                )
                self.stats.skipped_checkpoints += 1

        # Write to file
        self._write_json_file(output_file, export_data)

    def _export_multi_file(self, checkpoints: list[dict[str, Any]]) -> None:
        """Export checkpoints to multiple JSON files."""
        batch_num = 0

        for i in range(0, len(checkpoints), self.config.batch_size):
            batch = checkpoints[i : i + self.config.batch_size]
            batch_num += 1

            filename = f"checkpoints_export_batch_{batch_num:04d}.json"
            if self.compress:
                filename += ".gz"
            output_file = self.config.output_path / filename

            # Prepare batch data
            batch_data = {
                "metadata": {
                    "batch_number": batch_num,
                    "batch_size": len(batch),
                    "total_batches": (len(checkpoints) + self.config.batch_size - 1)
                    // self.config.batch_size,
                    "export_timestamp": datetime.now().isoformat(),
                },
                "checkpoints": [],
            }

            # Serialize batch
            for checkpoint in batch:
                try:
                    serialized = self._serialize_checkpoint(checkpoint)
                    batch_data["checkpoints"].append(serialized)
                    self.stats.exported_checkpoints += 1

                except Exception as e:
                    self.stats.errors.append(
                        f"Failed to serialize checkpoint {checkpoint.get('id')}: {e!s}",
                    )
                    self.stats.skipped_checkpoints += 1

            # Write batch file
            self._write_json_file(output_file, batch_data)

            # Report progress
            self._report_progress(
                min(i + self.config.batch_size, len(checkpoints)), len(checkpoints),
            )


    def _write_json_file(self, filepath: Path, data: dict[str, Any]) -> None:
        """Write JSON data to file with optional compression."""
        json_str = json.dumps(
            data, indent=2 if self.pretty_print else None, default=str,
        )
        self.stats.total_bytes += self._calculate_size(json_str)

        if self.compress:
            # Write compressed
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write(json_str)
        else:
            # Write plain text
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

    def export_by_thread(self, thread_id: str) -> ExportStats:
        """
        Export checkpoints for a specific thread.

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
