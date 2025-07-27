# Migration Scripts

This directory contains the migration tool for moving checkpoint data between different storage systems.

## Overview

The `migrate.py` script provides a flexible command-line interface for migrating checkpoints between:

**Sources:**
- PostgreSQL
- SQLite  
- Redis
- Unified Memory (Qdrant)

**Destinations:**
- Unified Memory (Qdrant)
- JSON export
- Parquet export

## Installation

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command Structure

```bash
python -m unified_checkpointer.migration_tools.scripts.migrate \
    --source <source_type> \
    --source-config '<json_config>' \
    --destination <dest_type> \
    --dest-config '<json_config>' \
    [options]
```

### Command Line Options

- `--source`: Source storage type (postgresql, sqlite, redis, unified)
- `--source-config`: JSON configuration for source connection
- `--destination`: Destination type (unified, json, parquet)
- `--dest-config`: JSON configuration for destination
- `--config`: Path to YAML configuration file (alternative to CLI args)
- `--dry-run`: Perform analysis without actual migration
- `--batch-size`: Number of records to process at once (default: 1000)
- `--verbose`: Enable detailed logging

### Examples

#### 1. PostgreSQL to Unified Memory

```bash
python -m unified_checkpointer.migration_tools.scripts.migrate \
    --source postgresql \
    --source-config '{"connection_string": "postgresql://user:pass@localhost/db", "table_name": "chat_history"}' \
    --destination unified \
    --dest-config '{"qdrant_url": "http://localhost:6333", "collection": "checkpoints"}' \
    --batch-size 500
```


#### 2. SQLite to JSON Export

```bash
python -m unified_checkpointer.migration_tools.scripts.migrate \
    --source sqlite \
    --source-config '{"connection_string": "sqlite:///checkpoints.db"}' \
    --destination json \
    --dest-config '{"output_dir": "./exports", "single_file": false}'
```

#### 3. Redis to Parquet

```bash
python -m unified_checkpointer.migration_tools.scripts.migrate \
    --source redis \
    --source-config '{"redis_url": "redis://localhost:6379", "key_prefix": "checkpoint:"}' \
    --destination parquet \
    --dest-config '{"output_path": "./data.parquet", "compression": "snappy"}'
```

#### 4. Using YAML Configuration

Create a `config.yaml` file:

```yaml
source_type: postgresql
source_config:
  connection_string: postgresql://user:pass@localhost/db
  table_name: chat_message_history
  
destination_type: unified
destination_config:
  qdrant_url: http://localhost:6333
  collection: migrated_checkpoints
  embedding_model: text-embedding-ada-002
  
filters:
  date_from: "2025-01-01"
  thread_pattern: "customer-*"
  
transformers:
  - add_metadata
  
batch_size: 1000
dry_run: false
```

Then run:
```bash
python -m unified_checkpointer.migration_tools.scripts.migrate --config config.yaml
```

## Configuration Options

### Source Configurations

#### PostgreSQL
```json
{
  "connection_string": "postgresql://user:pass@host:port/db",
  "table_name": "chat_message_history"
}
```

#### SQLite
```json
{
  "connection_string": "sqlite:///path/to/db.sqlite"
}
```

#### Redis
```json
{
  "redis_url": "redis://localhost:6379",
  "key_prefix": "checkpoint:",
  "decode_responses": true
}
```

#### Unified Memory (Qdrant)
```json
{
  "qdrant_url": "http://localhost:6333",
  "collection": "checkpoints",
  "api_key": "optional-api-key"
}
```

### Destination Configurations

#### Unified Memory
```json
{
  "qdrant_url": "http://localhost:6333",
  "collection": "new_checkpoints",
  "embedding_model": "text-embedding-ada-002"
}
```

#### JSON Export
```json
{
  "output_dir": "./exports",
  "single_file": false,
  "indent": 2
}
```

#### Parquet Export
```json
{
  "output_path": "./checkpoints.parquet",
  "compression": "snappy",
  "partition_by": ["thread_id"]
}
```

## Advanced Features

### Filters

Filter checkpoints during migration:

```yaml
filters:
  thread_id: "specific-thread-id"
  thread_pattern: "customer-*"  # Wildcard patterns
  date_from: "2025-01-01"
  date_to: "2025-12-31"
  exclude_threads:
    - "test-*"
    - "debug-*"
```

### Transformers

Apply transformations during migration:

```yaml
transformers:
  - add_metadata  # Adds migration timestamp
  - normalize_timestamps  # Converts to ISO format
  - custom_transformer  # Your custom transformer
```

### Dry Run

Always test migrations with `--dry-run` first:

```bash
python -m unified_checkpointer.migration_tools.scripts.migrate \
    --source postgresql \
    --source-config '...' \
    --destination unified \
    --dest-config '...' \
    --dry-run
```

This will:
- Count total records
- Sample data for analysis
- Check connectivity
- Estimate migration time
- Report potential issues

## Monitoring

The migration tool provides:
- Progress updates every N records
- Error tracking and reporting
- Final statistics summary
- Detailed logging with `--verbose`

## Error Handling

The tool includes:
- Automatic retries for transient errors
- Batch-level error isolation
- Detailed error logging
- Graceful shutdown on critical errors

## Performance Tips

1. **Batch Size**: Adjust based on record size and network
   - Large records: Use smaller batches (100-500)
   - Small records: Use larger batches (1000-5000)

2. **Parallel Processing**: For large migrations, run multiple instances with different filters

3. **Network**: Ensure source and destination are network-accessible

4. **Memory**: Monitor memory usage for large batch sizes

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify connection strings
   - Check network accessibility
   - Ensure services are running

2. **Authentication Failures**
   - Check credentials in connection strings
   - Verify API keys for Qdrant

3. **Memory Issues**
   - Reduce batch size
   - Use filters to process subsets

4. **Data Format Errors**
   - Check source data compatibility
   - Verify serialization format

### Debug Mode

Enable verbose logging:
```bash
python -m unified_checkpointer.migration_tools.scripts.migrate \
    --config config.yaml \
    --verbose
```

## See Also

- [Example Migrations](../examples/example_migrations.py)
- [Migration Configuration Examples](../examples/)
- [Unified Checkpointer Documentation](../../README.md)
