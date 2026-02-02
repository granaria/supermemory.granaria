# Local Supermemory

A fully local, privacy-first memory system for Claude AI with semantic search and automatic profile generation.

**Drop-in replacement for Supermemory cloud** – all your data stays on your machine.

## Features

- 🔒 **100% Local** – No cloud, no external APIs for memory storage
- 🔍 **Semantic Search** – ChromaDB-powered vector similarity search
- 🧠 **Profile Generation** – Automatic user profile via local Ollama LLM
- 📁 **Project Support** – Organize memories by project
- 🔌 **MCP Server** – Direct integration with Claude Desktop
- 📦 **Zero Config** – Works out of the box

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/local-supermemory.git
cd local-supermemory
pip install -e .
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for profile generation)

## Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "local-supermemory": {
      "command": "python3",
      "args": ["-m", "local_supermemory.server"]
    }
  }
}
```

Restart Claude Desktop.

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory` | Save or forget information (`action`: save/forget) |
| `recall` | Semantic search with optional profile summary |
| `list_projects` | List all memory projects |
| `stats` | Storage statistics |
| `whoami` | Generate user profile from memories |

## Usage Examples

**Save a memory:**
```
Remember that my email is user@example.com
```

**Search memories:**
```
What do you know about my trading setup?
```

**Project-scoped memories:**
```
Save to project "work": Meeting notes from today...
```

## Data Storage

All data is stored locally in `~/.local-supermemory/`:

```
~/.local-supermemory/
├── memories.db      # SQLite: metadata, projects, profile cache
└── chroma/          # ChromaDB: vector embeddings
```

## Profile Generation

If [Ollama](https://ollama.ai) is running locally, the `whoami` tool generates an intelligent profile summary from your memories. Configure the model in `profile.py`:

```python
OLLAMA_MODEL = "qwen2.5:32b"  # or any model you have
```

Without Ollama, a simple keyword-based profile is generated.

## Migration from Supermemory Cloud

Create a `migrate.py` script to export your cloud memories:

```python
# Example migration - customize for your needs
from local_supermemory.store import MemoryStore

store = MemoryStore()
memories = [
    "User email is user@example.com",
    "User prefers Python for coding",
    # ... your memories
]

for m in memories:
    store.save(m)

print(f"Migrated {len(memories)} memories")
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐
│  Claude Desktop │────▶│  MCP Server  │
└─────────────────┘     └──────┬───────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
              ┌─────▼─────┐        ┌──────▼──────┐
              │  ChromaDB │        │   SQLite    │
              │ (vectors) │        │ (metadata)  │
              └───────────┘        └─────────────┘
                    │
              ┌─────▼─────┐
              │  Ollama   │ (optional)
              │ (profile) │
              └───────────┘
```

## License

MIT License – see [LICENSE](LICENSE)

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

Inspired by [Supermemory](https://supermemory.ai) – this project provides a local alternative for privacy-conscious users.
