"""Base exporter for checkpoint data migration."""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExportConfig:
    """Configuration for checkpoint export."""

    # Filter options
    thread_ids: set[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    limit: int | None = None

    # Output options
    output_path: Path = field(default_factory=lambda: Path("./export"))
    batch_size: int = 100
    include_metadata: bool = True

    # Progress tracking
    progress_callback: Callable[[int, int], None] | None = None

    def __post_init__(self):
        """Ensure output path is a Path object."""
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)


@dataclass
class ExportStats:
    """Statistics about the export operation."""

    total_checkpoints: int = 0
    exported_checkpoints: int = 0
    skipped_checkpoints: int = 0
    total_bytes: int = 0
    export_duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class BaseExporter(ABC):
    """Abstract base class for checkpoint exporters."""

    def __init__(self, client, config: ExportConfig | None = None) -> None:
        """
        Initialize the exporter.

        Args:
            client: UnifiedMemoryClient instance
            config: Export configuration
        """
        self.client = client
        self.config = config or ExportConfig()
        self.stats = ExportStats()

    @abstractmethod
    def export_checkpoints(self) -> ExportStats:
        """
        Export checkpoints according to configuration.

        Returns:
            Export statistics
        """

    def _filter_checkpoints(
        self, checkpoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Apply filters to checkpoints.

        Args:
            checkpoints: List of checkpoint documents

        Returns:
            Filtered list of checkpoints
        """
        filtered = checkpoints

        # Filter by thread IDs
        if self.config.thread_ids:
            filtered = [
                cp for cp in filtered if cp.get("thread_id") in self.config.thread_ids
            ]

        # Filter by date range
        if self.config.start_date or self.config.end_date:
            filtered = self._filter_by_date(filtered)
        # Apply limit if specified
        if self.config.limit and len(filtered) > self.config.limit:
            filtered = filtered[: self.config.limit]

        return filtered

    def _filter_by_date(
        self, checkpoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter checkpoints by date range."""
        result = []

        for cp in checkpoints:
            # Extract timestamp from checkpoint
            ts = cp.get("ts")
            if not ts:
                continue

            try:
                # Handle different timestamp formats
                if isinstance(ts, str):
                    checkpoint_date = datetime.fromisoformat(ts)
                elif isinstance(ts, (int, float)):
                    checkpoint_date = datetime.fromtimestamp(ts)
                else:
                    checkpoint_date = ts

                # Check date range
                if self.config.start_date and checkpoint_date < self.config.start_date:
                    continue
                if self.config.end_date and checkpoint_date > self.config.end_date:
                    continue

                result.append(cp)

            except Exception as e:
                self.stats.errors.append(
                    f"Date parsing error for checkpoint {cp.get('id')}: {e!s}",
                )

        return result

    def _serialize_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """
        Serialize checkpoint data for export.

        Args:
            checkpoint: Raw checkpoint document

        Returns:
            Serialized checkpoint data
        """
        serialized = {
            "id": checkpoint.get("id"),
            "thread_id": checkpoint.get("thread_id"),
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
            "type": checkpoint.get("type"),
            "checkpoint": checkpoint.get("checkpoint"),
            "metadata": checkpoint.get("metadata", {}),
        }

        # Add timestamp in ISO format
        ts = checkpoint.get("ts")
        if ts:
            if isinstance(ts, str):
                serialized["timestamp"] = ts
            elif isinstance(ts, (int, float)):
                serialized["timestamp"] = datetime.fromtimestamp(ts).isoformat()
            else:
                serialized["timestamp"] = (
                    ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                )

        return serialized

    def _report_progress(self, current: int, total: int) -> None:
        """Report export progress if callback is configured."""
        if self.config.progress_callback:
            self.config.progress_callback(current, total)

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        if isinstance(data, (dict, list)):
            return len(json.dumps(data).encode("utf-8"))
        if isinstance(data, str):
            return len(data.encode("utf-8"))
        return len(str(data).encode("utf-8"))

    def _create_export_metadata(self) -> dict[str, Any]:
        """Create metadata about the export operation."""
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "exporter_class": self.__class__.__name__,
            "export_config": {
                "thread_ids": list(self.config.thread_ids)
                if self.config.thread_ids
                else None,
                "start_date": self.config.start_date.isoformat()
                if self.config.start_date
                else None,
                "end_date": self.config.end_date.isoformat()
                if self.config.end_date
                else None,
                "limit": self.config.limit,
                "batch_size": self.config.batch_size,
            },
            "stats": {
                "total_checkpoints": self.stats.total_checkpoints,
                "exported_checkpoints": self.stats.exported_checkpoints,
                "skipped_checkpoints": self.stats.skipped_checkpoints,
                "total_bytes": self.stats.total_bytes,
                "export_duration_seconds": self.stats.export_duration_seconds,
                "errors_count": len(self.stats.errors),
            },
        }

        if self.stats.errors:
            metadata["errors"] = self.stats.errors[:10]  # Limit to first 10 errors

        return metadata