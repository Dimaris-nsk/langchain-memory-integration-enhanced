# Migration Tools

Complete migration toolkit for checkpoint data between different storage systems.

## Features

- **Multiple Source Support**: PostgreSQL, SQLite, Redis, UnifiedCheckpointer
- **Multiple Destination Support**: UnifiedCheckpointer, JSON, Parquet
- **Batch Processing**: Efficient migration of large datasets
- **Progress Tracking**: Real-time progress updates
- **Dry Run Mode**: Preview migration without making changes
- **Filters**: Migrate only specific data
- **Transformers**: Modify data during migration

## Directory Structure

```
migration_tools/
├── importers/           # Source importers
│   ├── base_importer.py
│   ├── postgresql_importer.py
│   ├── sqlite_importer.py
│   └── redis_importer.py
├── exporters/           # Destination exporters
│   ├── base_exporter.py
│   ├── json_exporter.py
│   └── parquet_exporter.py
├── scripts/             # Migration scripts
│   └── migrate.py
├── examples/            # Configuration examples
│   ├── postgres_to_unified.yaml
│   └── unified_to_json.yaml
└── tests/              # Test files

## Quick Start

### 1. Command Line Usage

```bash
# Migrate from PostgreSQL to UnifiedCheckpointer
python migrate.py \
  --source postgresql \
  --source-config '{"connection_string": "postgresql://localhost/db"}' \
  --destination unified \
  --dest-config '{"url": "http://localhost:6333", "collection": "checkpoints"}' \
  --batch-size 1000

# Dry run to preview migration
python migrate.py \
  --config examples/postgres_to_unified.yaml \
  --dry-run

# Export to JSON with progress tracking
python migrate.py \
  --source unified \
  --source-config '{"url": "http://localhost:6333", "collection": "checkpoints"}' \
  --destination json \
  --dest-config '{"output_path": "export.json"}' \
  --verbose
```

### 2. Configuration File Usage

Create a YAML configuration file:

```yaml
source_type: postgresql
source_config:
  connection_string: "postgresql://user:pass@host:5432/db"

destination_type: unified
destination_config:
  url: "http://localhost:6333"
  collection: "migrated_checkpoints"

batch_size: 500
filters:
  date_from: "2025-01-01T00:00:00"
```

Run migration:
```bash
python migrate.py --config migration_config.yaml
```

## Importers

### PostgreSQL Importer
- Imports from ChatMessageHistory tables
- Supports custom table names
- Handles timestamp conversion

### SQLite Importer
- Imports from SQLite databases
- Auto-detects message format
- Supports both file paths and memory databases

### Redis Importer
- Imports from Redis lists, hashes, sorted sets
- Pattern-based key discovery
- Supports Redis Cluster

## Exporters

### JSON Exporter
- Human-readable JSON format
- Optional pretty printing
- Configurable embedding inclusion

### Parquet Exporter
- Columnar storage format
- Efficient for analytics
- Supports partitioning by thread/date
- Multiple compression options

## Filters

Configure filters to migrate only specific data:

```yaml
filters:
  thread_id: "session-123"          # Single thread
  thread_ids: ["s1", "s2", "s3"]   # Multiple threads
  date_from: "2025-01-01"          # Date range
  date_to: "2025-07-24"
  metadata_filter:                  # Custom metadata
    user_id: "user-456"
```

## Transformers

Apply transformations during migration:

```yaml
transformers:
  - add_metadata      # Add migration timestamp
  - anonymize_pii     # Remove personal information
  - compress_data     # Compress large payloads
```

## Performance Tips

1. **Batch Size**: Adjust based on your data size and memory
   - Small records: 1000-5000 per batch
   - Large records: 100-500 per batch

2. **Parallel Processing**: For large migrations, split by thread_id

3. **Network Optimization**: Run migration close to data source

4. **Memory Management**: Monitor memory usage for large exports

## Error Handling

The migration tool includes:
- Automatic retry on transient failures
- Detailed error logging
- Option to continue on errors
- Recovery from interruptions

## Testing

Run tests:
```bash
pytest tests/
```

## Examples

See the `examples/` directory for more configuration examples:
- `postgres_to_unified.yaml` - PostgreSQL to Qdrant migration
- `unified_to_json.yaml` - Export to JSON format
- `redis_to_parquet.yaml` - Redis to Parquet export
- `sqlite_migration.yaml` - SQLite with filters

## Troubleshooting

### Connection Issues
- Verify connection strings
- Check network connectivity
- Ensure services are running

### Performance Issues
- Reduce batch size
- Enable verbose logging
- Check resource usage

### Data Issues
- Validate source data format
- Check filter configuration
- Review transformation logs