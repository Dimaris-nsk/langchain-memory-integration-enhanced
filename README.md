# 🚀 Langgraph + Qdrant Memory Agent MCP Server

> **A production-ready memory management system with semantic search and conversation history**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Langgraph](https://img.shields.io/badge/langgraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/qdrant-vector_db-purple.svg)](https://qdrant.tech/)
[![MCP](https://img.shields.io/badge/MCP-server-orange.svg)](https://modelcontextprotocol.io/)

---

## 📋 Table of Contents

- [🎯 Philosophy & Goals](#-philosophy--goals)
- [✨ Features](#-features)
- [🛠️ Technology Stack](#-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📚 Usage](#-usage)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🕰️ Legacy Section](#-legacy-section)

---

## 🎯 Philosophy & Goals

This project is not just another MCP server. It's a **teaching tool** and **reference implementation** designed to demonstrate best practices in software development.

### Core Principles:
- **🎨 Clean Architecture**: Beautiful, maintainable code structure
- **📚 Educational Value**: Learn by example how to build quality software
- **⚡ Production Ready**: Not a prototype, but a real-world solution
- **🔍 Transparency**: Every decision is documented and explained

### What Makes This Project Special:
- **Exemplary Quality**: A "good, solid, quality project" that others can learn from
- **Real-World Ready**: Designed for actual production use, not just demos
- **Community Focused**: Built to be understood, extended, and improved by others
- **Best Practices**: Demonstrates modern Python development standards

---

## ✨ Features

### 🧠 Memory Management
- **Semantic Search**: BGE-M3 embeddings for intelligent memory retrieval
- **Multi-language Support**: Automatic language detection and filtering
- **Tag System**: Organize memories with flexible tagging
- **Metadata Rich**: Store context with every memory

### 💬 Conversation History
- **Full Context Tracking**: Never lose important conversation details
- **Hybrid Retrieval**: Combines recent and semantically relevant messages
- **Smart Summarization**: MVP concatenation with future LLM support
- **Session Management**: Organized by user and session IDs

### 🔍 Advanced Search
- **Natural Language Queries**: Search like you think
- **Time-based Filtering**: "Show me memories from last week"
- **Faceted Search**: Drill down by tags, language, metadata
- **Search Analytics**: Understand what users are looking for

### 🔄 Import/Export
- **Full Backup Support**: Export entire collections to JSON
- **Selective Export**: Export by user, session, or criteria
- **Migration Ready**: Import from other systems
- **Data Portability**: Your data, your control

---

## 🛠️ Technology Stack

### Core Technologies:
- **🐍 Python 3.11+**: Modern Python with type hints
- **🔗 Langgraph**: For building stateful agents
- **🎯 Qdrant**: High-performance vector database
- **🤖 OpenAI**: For embeddings and future LLM features
- **🔌 MCP (Model Context Protocol)**: Standardized AI tool interface

### Key Libraries:
- **FastMCP**: MCP server implementation
- **BGE-M3**: State-of-the-art multilingual embeddings
- **Pydantic**: Data validation and settings
- **HTTPX**: Modern async HTTP client
- **Rich**: Beautiful terminal output

---

## 📁 Project Structure

```
langraph-qdrant-memory-agent/
├── 📋 metaplan/                    # Project planning and progress
│   ├── memory-agent-metaplan.md   # Detailed development plan
│   └── memory-agent-metaplan-log.md # Progress tracking
├── 🧠 unified_memory/             # Core memory functionality
│   ├── __init__.py
│   ├── core.py                    # Main memory operations
│   ├── search.py                  # Advanced search features
│   ├── conversation.py            # Conversation management
│   ├── language.py                # Multi-language support
│   └── import_export.py           # Data portability
├── 🔗 unified_checkpointer/       # Langgraph integration
│   ├── __init__.py
│   ├── checkpointer.py            # State persistence
│   ├── store.py                   # Key-value storage
│   └── models.py                  # Data models
├── 🧪 tests/                      # Comprehensive test suite
│   ├── test_memory.py
│   ├── test_search.py
│   ├── test_conversation.py
│   └── test_checkpointer.py
├── 📚 docs/                       # Documentation
│   ├── architecture.md
│   ├── api.md
│   └── examples.md
├── 🔧 scripts/                    # Utility scripts
│   ├── setup_qdrant.py
│   └── migrate_data.py
├── server.py                      # MCP server entry point
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # You are here!
```

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.11+**
   ```bash
   python --version  # Should show 3.11 or higher
   ```

2. **Qdrant** (Choose one):
   - Docker (Recommended):
     ```bash
     docker run -p 6333:6333 qdrant/qdrant
     ```
   - Or Qdrant Cloud: [Sign up here](https://cloud.qdrant.io/)

3. **OpenAI API Key** (for embeddings):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dimaris-nsk/langraph-qdrant-memory-agent.git
   cd langraph-qdrant-memory-agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Initialize Qdrant**:
   ```bash
   python scripts/setup_qdrant.py
   ```

### Running the Server

```bash
# Standard mode
python server.py

# Debug mode with rich output
python server.py --debug

# Custom port
python server.py --port 8080
```

---

## 📚 Usage

### Basic Memory Operations

```python
# Store a memory
store_memory({
    "content": "Important meeting notes from project kickoff",
    "tags": ["meeting", "project-x", "important"],
    "metadata": {"date": "2025-01-29", "attendees": ["Alice", "Bob"]}
})

# Retrieve memories
results = retrieve_memory({
    "query": "project kickoff meeting",
    "tags": ["important"],
    "limit": 5
})

# Search by tags
memories = search_by_tags({
    "tags": ["meeting", "important"]
})
```

### Conversation Management

```python
# Store conversation turn
store_conversation_turn({
    "session_id": "session-123",
    "user_id": "user-456",
    "human_message": "Tell me about the weather",
    "ai_message": "I'll help you with weather information..."
})

# Get conversation summary
summary = get_conversation_summary({
    "session_id": "session-123",
    "user_id": "user-456",
    "max_tokens": 2000
})

# Search conversation history
results = search_conversation_history({
    "session_id": "session-123",
    "user_id": "user-456",
    "semantic_query": "weather discussion",
    "k": 5
})
```

### Advanced Search

```python
# Natural language search
results = advanced_search({
    "query": "meetings from last week about project X",
    "include_facets": True
})

# Parse natural query
parsed = parse_natural_query({
    "query": "show me important Python code from yesterday"
})
# Returns: {"tags": ["important", "python"], "time_range": "yesterday", ...}
```

### Import/Export

```python
# Export collection
export_collection({
    "output_path": "/backup/memory_2025-01-29.json",
    "with_vectors": True
})

# Import collection
import_collection({
    "input_path": "/backup/memory_2025-01-29.json",
    "merge_payload": True,
    "skip_existing": True
})
```

---

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=unified_memory --cov=unified_checkpointer

# Specific test file
pytest tests/test_memory.py

# With verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **E2E Tests**: Full workflow testing
- **Performance Tests**: Ensure scalability

---

## 🤝 Contributing

We welcome contributions! This project is designed to be a learning experience for everyone.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Guidelines

- **Code Quality**: Follow PEP 8 and use type hints
- **Tests**: Add tests for new features
- **Documentation**: Update docs for changes
- **Commits**: Use conventional commits format

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run code formatter
black .

# Run linter
flake8

# Run type checker
mypy .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🕰️ Legacy Section

### Project History

This project evolved from `langchain-memory-integration-enhanced`, originally created as a test task to showcase Langchain skills. What started as a demonstration became a full-fledged production system.

### Evolution Timeline:
1. **Initial Phase**: Basic Langchain memory integration
2. **Enhancement Phase**: Added Qdrant vector storage
3. **Production Phase**: Rebuilt with Langgraph for stateful agents
4. **Current Phase**: MCP server with comprehensive features

### Original Vision

From the creator:
> "Я хочу донести до людей что правильно выстроенный воркфлоу когда все работает слаженно и создается действительно правильный качественный код по настоящему качественное ПО..."

Translation: "I want to show people that a properly structured workflow, where everything works harmoniously, creates truly correct, quality code and genuinely quality software..."

This vision drives every decision in this project.

### Lessons Learned

1. **Start with solid architecture** - It pays off in the long run
2. **Document as you go** - Future you will thank present you
3. **Test everything** - Quality requires verification
4. **Think production from day one** - Prototypes often become products

---

## 🎯 Why This Project Matters

In a world of quick demos and throwaway code, this project stands as an example of what software development should be:

- **Thoughtful**: Every line of code has a purpose
- **Maintainable**: Easy to understand and extend
- **Educational**: Learn best practices by example
- **Production-Ready**: Not just a demo, but a real tool

Whether you're learning to code or teaching others, this project demonstrates that quality matters, process matters, and doing things right the first time is always worth it.

---

*Built with ❤️ and attention to quality*

*"Quality is not an act, it is a habit." - Aristotle*