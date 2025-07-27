# LangChain Memory Integration Enhanced - Deployment Guide

## 🚀 Overview

This guide covers deployment strategies, configurations, and best practices for deploying LangChain Memory Integration Enhanced in production environments.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8+ (3.11+ recommended)
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Storage**: Depends on vector store backend
- **Network**: Low-latency connection to vector store

### Required Services

Depending on your configuration, you'll need:

- **Vector Store**: One of:
  - Qdrant (recommended for production)
  - Chroma (good for development)
  - Pinecone (cloud-based option)
  - Weaviate (enterprise option)
- **Cache** (optional): Redis 6.0+
- **Monitoring** (recommended): Prometheus + Grafana

## 🔧 Installation

### Production Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from PyPI (when available)
pip install langchain-memory-integration-enhanced

# Or install from source
git clone https://github.com/Dimaris-nsk/langchain-memory-integration-enhanced.git
cd langchain-memory-integration-enhanced
pip install -e .
```

### Docker Installation

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANGCHAIN_VECTOR_STORE_TYPE=qdrant

CMD ["python", "-m", "your_application"]
```

## 🎯 Deployment Strategies

### 1. Standalone Deployment

Best for: Development, small-scale applications

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      - LANGCHAIN_VECTOR_STORE_TYPE=qdrant
      - LANGCHAIN_QDRANT_URL=http://qdrant:6333
      - LANGCHAIN_CACHE_ENABLED=true
      - LANGCHAIN_REDIS_URL=redis://redis:6379
    depends_on:
      - qdrant
      - redis

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  qdrant_data:
  redis_data:
```

### 2. Kubernetes Deployment

Best for: Production, scalable applications

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-memory-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-memory
  template:
    metadata:
      labels:
        app: langchain-memory
    spec:
      containers:
      - name: app
        image: your-registry/langchain-memory:latest
        env:
        - name: LANGCHAIN_VECTOR_STORE_TYPE
          value: "qdrant"
        - name: LANGCHAIN_QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: LANGCHAIN_CACHE_ENABLED
          value: "true"
        - name: LANGCHAIN_REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 3. Serverless Deployment (AWS Lambda)

Best for: Event-driven, cost-effective scaling

```python
# lambda_handler.py
import os
from langchain_memory_integration import EnhancedMemoryCheckpointer, Settings

# Initialize outside handler for connection reuse
settings = Settings(
    vector_store_type=os.getenv("VECTOR_STORE_TYPE", "qdrant"),
    qdrant_url=os.getenv("QDRANT_URL"),
    cache_enabled=False  # Lambda has limited local storage
)
checkpointer = EnhancedMemoryCheckpointer(settings)

def lambda_handler(event, context):
    operation = event.get("operation")
    
    if operation == "save":
        result = checkpointer.save_sync(
            memory_data=event["memory_data"],
            tags=event.get("tags", []),
            metadata=event.get("metadata", {})
        )
        return {"statusCode": 200, "body": {"memory_id": result}}
    
    elif operation == "load":
        result = checkpointer.load_sync(
            memory_id=event.get("memory_id"),
            tags=event.get("tags"),
            limit=event.get("limit", 10)
        )
        return {"statusCode": 200, "body": result}
    
    return {"statusCode": 400, "body": "Invalid operation"}
```

## ⚙️ Configuration

### Environment-Based Configuration

```bash
# .env.production
# Vector Store Configuration
LANGCHAIN_VECTOR_STORE_TYPE=qdrant
LANGCHAIN_QDRANT_URL=https://qdrant.production.example.com:6333
LANGCHAIN_QDRANT_API_KEY=your-secure-api-key
LANGCHAIN_QDRANT_COLLECTION=production_memories

# Cache Configuration
LANGCHAIN_CACHE_ENABLED=true
LANGCHAIN_REDIS_URL=redis://:password@redis.production.example.com:6379/0
LANGCHAIN_CACHE_TTL=3600

# Resilience Configuration
LANGCHAIN_MAX_RETRIES=5
LANGCHAIN_RETRY_DELAY=2.0
LANGCHAIN_CIRCUIT_BREAKER_THRESHOLD=10
LANGCHAIN_CIRCUIT_BREAKER_TIMEOUT=120

# Performance Configuration
LANGCHAIN_BATCH_SIZE=500
LANGCHAIN_CONNECTION_POOL_SIZE=20
LANGCHAIN_REQUEST_TIMEOUT=30

# Monitoring
LANGCHAIN_METRICS_ENABLED=true
LANGCHAIN_METRICS_PORT=9090
```

### Configuration File Approach

```yaml
# config.production.yaml
vector_store:
  type: qdrant
  qdrant:
    url: https://qdrant.production.example.com:6333
    api_key: ${QDRANT_API_KEY}
    collection: production_memories
    timeout: 30
    retries: 3

cache:
  enabled: true
  type: redis
  redis:
    url: ${REDIS_URL}
    ttl: 3600
    max_connections: 50

resilience:
  circuit_breaker:
    threshold: 10
    timeout: 120
    half_open_requests: 3
  retry:
    max_attempts: 5
    base_delay: 2.0
    max_delay: 60.0
    jitter: true

performance:
  batch_size: 500
  connection_pool_size: 20
  async_workers: 10
```

## 🔐 Security Best Practices

### 1. API Key Management

```python
# Use environment variables
api_key = os.getenv("QDRANT_API_KEY")

# Use secrets management service
from aws_secretsmanager import get_secret
api_key = get_secret("langchain/qdrant/api_key")

# Never hardcode keys
# BAD: api_key = "your-actual-key-here"
```

### 2. Network Security

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: langchain-memory-netpol
spec:
  podSelector:
    matchLabels:
      app: langchain-memory
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### 3. Data Encryption

```python
# Enable TLS for connections
settings = Settings(
    qdrant_url="https://qdrant.example.com:6333",  # HTTPS
    qdrant_tls_verify=True,
    redis_url="rediss://redis.example.com:6379",    # Redis with TLS
    redis_ssl_cert_reqs="required"
)
```

## 📊 Monitoring & Observability

### 1. Health Checks

```python
# Flask example
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/health")
async def health():
    health_status = await checkpointer.health_check()
    status_code = 200 if health_status.status == "HEALTHY" else 503
    return jsonify(health_status.dict()), status_code

@app.route("/ready")
async def ready():
    # Check if service is ready to handle requests
    try:
        await checkpointer.load("test-id")
        return jsonify({"status": "ready"}), 200
    except:
        return jsonify({"status": "not ready"}), 503
```

### 2. Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
memory_operations = Counter(
    'langchain_memory_operations_total',
    'Total number of memory operations',
    ['operation', 'status']
)

operation_duration = Histogram(
    'langchain_memory_operation_duration_seconds',
    'Duration of memory operations',
    ['operation']
)

active_connections = Gauge(
    'langchain_memory_active_connections',
    'Number of active connections to vector store'
)

# Start metrics server
start_http_server(9090)
```

### 3. Logging Configuration

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure structured logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(timestamp)s %(level)s %(name)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logHandler.setFormatter(formatter)

logger = logging.getLogger('langchain_memory')
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Add context to logs
logger.info("Memory operation completed", extra={
    "operation": "save",
    "memory_id": "abc123",
    "duration_ms": 150,
    "tags": ["session-123"],
    "user_id": "user-456"
})
```

## 🔄 Migration & Upgrades

### Zero-Downtime Migration

```python
# migration_strategy.py
from langchain_memory_integration.migration_tools import MigrationEngine

async def zero_downtime_migration():
    # 1. Set up dual-write mode
    old_checkpointer = EnhancedMemoryCheckpointer(old_settings)
    new_checkpointer = EnhancedMemoryCheckpointer(new_settings)
    
    # 2. Start background migration
    migration_engine = MigrationEngine(
        source_settings=old_settings,
        target_settings=new_settings
    )
    
    migration_task = asyncio.create_task(
        migration_engine.migrate(
            batch_size=1000,
            parallel_workers=4
        )
    )
    
    # 3. Dual-write during migration
    async def dual_write_save(data, **kwargs):
        # Write to both systems
        old_id = await old_checkpointer.save(data, **kwargs)
        new_id = await new_checkpointer.save(data, **kwargs)
        return new_id
    
    # 4. Verify migration completion
    await migration_task
    verification = await migration_engine.verify_migration()
    
    # 5. Switch to new system
    if verification.success:
        return new_checkpointer
    else:
        raise MigrationError(f"Migration failed: {verification.errors}")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Connection Timeouts

```python
# Increase timeout settings
settings = Settings(
    qdrant_timeout=60,  # 60 seconds
    redis_connection_timeout=10,
    max_retries=5
)
```

#### 2. Memory Issues

```bash
# Monitor memory usage
docker stats

# Increase container limits
docker run -m 4g ...

# Or in Kubernetes
resources:
  limits:
    memory: "4Gi"
```

#### 3. Performance Degradation

```python
# Enable connection pooling
settings = Settings(
    connection_pool_size=50,
    connection_pool_max_overflow=10
)

# Optimize batch sizes
settings.batch_size = 1000  # Adjust based on your data
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed error messages
settings = Settings(
    debug_mode=True,
    verbose_errors=True
)
```

## 📈 Performance Tuning

### 1. Connection Pooling

```python
# Optimize for high throughput
settings = Settings(
    connection_pool_size=100,
    connection_pool_max_overflow=20,
    connection_pool_timeout=30
)
```

### 2. Batch Processing

```python
# Tune batch sizes based on memory and network
settings = Settings(
    batch_size=5000,  # Large batches for bulk operations
    batch_timeout=60,
    parallel_batch_workers=10
)
```

### 3. Caching Strategy

```python
# Configure aggressive caching
settings = Settings(
    cache_enabled=True,
    cache_ttl=7200,  # 2 hours
    cache_max_size=10000,  # Max items in cache
    cache_eviction_policy="lru"
)
```

## 🔗 Integration Examples

### With FastAPI

```python
# main.py
from fastapi import FastAPI, HTTPException
from langchain_memory_integration import EnhancedMemoryCheckpointer

app = FastAPI()
checkpointer = EnhancedMemoryCheckpointer()

@app.post("/memory")
async def save_memory(data: dict):
    try:
        memory_id = await checkpointer.save(data)
        return {"memory_id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{memory_id}")
async def get_memory(memory_id: str):
    try:
        memory = await checkpointer.load(memory_id)
        return memory
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="Memory not found")
```

### With Celery

```python
# tasks.py
from celery import Celery
from langchain_memory_integration import EnhancedMemoryCheckpointer

app = Celery('memory_tasks')
checkpointer = EnhancedMemoryCheckpointer()

@app.task
def save_memory_async(data, tags=None, metadata=None):
    return checkpointer.save_sync(data, tags=tags, metadata=metadata)

@app.task
def migrate_memories_batch(memory_ids):
    for memory_id in memory_ids:
        # Process migration
        pass
```

## 📚 Additional Resources

- [Architecture Overview](./ARCHITECTURE.md)
- [API Reference](./API_REFERENCE.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
- [Performance Tuning](./PERFORMANCE_TUNING.md)

---

*Last updated: 2025-07-28*