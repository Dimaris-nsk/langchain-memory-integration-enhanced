# LangChain Memory Integration Enhanced

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://img.shields.io/badge/code%20quality-SonarQube-green.svg)](http://localhost:9000)
[![Test Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen.svg)](https://github.com/Dimaris-nsk/langchain-memory-integration-enhanced)

A high-quality, production-ready memory management system for LangChain applications with automatic memory persistence, resilience patterns, and clean architecture implementation.

## 🌟 Features

- **4 Memory Types** - Drop-in replacements for LangChain memory classes:
  - `UnifiedConversationBufferMemory` - Full conversation history with automatic persistence
  - `UnifiedConversationSummaryMemory` - Compressed summaries for long conversations
  - `UnifiedEntityMemory` - Entity extraction and relationship tracking
  - `UnifiedKnowledgeGraphMemory` - Graph-based memory with semantic connections

- **Automatic Memory Management**:
  - Auto-saves conversation context without manual intervention
  - Intelligent memory routing based on content type
  - Configurable retention policies and cleanup strategies

- **Resilience & Reliability**:
  - Circuit breaker pattern for fault tolerance
  - Graceful degradation when services unavailable
  - Retry mechanisms with exponential backoff
  - Health checks and monitoring

- **Clean Architecture**:
  - Hexagonal architecture with clear boundaries
  - Dependency injection for testability
  - Interface-based design for extensibility
  - Full type hints and documentation

## 🏗️ Architecture

The project follows Clean Architecture principles with clear separation of concerns:

```
├── implementation/
│   ├── core/               # Core business logic
│   │   ├── interfaces/     # Abstract interfaces
│   │   ├── entities/       # Domain entities
│   │   └── use_cases/      # Business use cases
│   ├── adapters/           # External adapters
│   │   ├── qdrant/         # Qdrant vector store
│   │   └── openai/         # OpenAI embeddings
│   ├── unified_checkpointer/  # Memory persistence
│   ├── resilience/         # Fault tolerance patterns
│   └── compatibility/      # LangChain compatibility layer
├── tests/                  # Comprehensive test suite
└── docs/                   # Documentation
```

## 📋 Requirements

- Python 3.9+
- Qdrant (local or cloud)
- OpenAI API key (for embeddings)
- Optional: LangGraph for advanced orchestration

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Dimaris-nsk/langchain-memory-integration-enhanced.git
cd langchain-memory-integration-enhanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## 💻 Quick Start

### Basic Usage

```python
from langchain_memory_enhanced import UnifiedConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Initialize memory with automatic persistence
memory = UnifiedConversationBufferMemory(
    collection_name="my_conversations",
    qdrant_url="http://localhost:6333"
)

# Use with LangChain as usual
llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Memory is automatically saved after each interaction
response = conversation.predict(input="Hi there!")
```

### Advanced Configuration

```python
from langchain_memory_enhanced import UnifiedMemoryConfig, UnifiedConversationSummaryMemory

config = UnifiedMemoryConfig(
    qdrant_url="http://localhost:6333",
    collection_name="production_memory",
    embedding_model="text-embedding-3-small",
    max_memory_size=1000,
    cleanup_strategy="sliding_window",
    enable_monitoring=True,
    circuit_breaker_threshold=5,
    retry_attempts=3
)

memory = UnifiedConversationSummaryMemory(config=config)
```

## 📖 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and patterns
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Migration Guide](migration_tools/README.md) - Migrating from other memory systems

## 🧪 Testing

The project maintains >80% test coverage with comprehensive unit and integration tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=implementation --cov-report=html

# Run specific test suite
pytest tests/test_unified_checkpointer.py
```

## 🛠️ Development

### Code Quality

We use several tools to maintain high code quality:

```bash
# Run linting
ruff check implementation/

# Format code
ruff format implementation/

# Type checking
mypy implementation/

# SonarQube analysis
sonar-scanner
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## 🔧 MCP Server Integration

This project can be used as an MCP (Model Context Protocol) server for enhanced memory capabilities in AI applications:

```python
from langchain_memory_mcp import MemoryMCPServer

server = MemoryMCPServer(
    memory_types=["buffer", "summary", "entity", "knowledge_graph"],
    auto_save=True
)
server.start()
```

## 📊 Performance

- **Latency**: <50ms for memory operations
- **Throughput**: 1000+ operations/second
- **Memory efficiency**: Automatic cleanup and compression
- **Scalability**: Horizontal scaling with Qdrant cluster

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- Qdrant team for the vector database
- LangGraph for orchestration capabilities
- All contributors who helped improve this project

---

**Note**: This is a clean architecture implementation focused on production quality and maintainability. For questions or support, please open an issue or contact the maintainers.