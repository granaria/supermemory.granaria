# Local Supermemory

A fully local, privacy-first memory system for Claude AI with semantic search, knowledge graph, RAG-based Q&A, progressive disclosure, and automatic profile generation.

**Drop-in replacement for Supermemory cloud** — all your data stays on your machine.

## Features

- 🔒 **100% Local** — No cloud, no external APIs for memory storage
- 🔍 **Semantic Search** — ChromaDB-powered vector similarity search with sentence-level chunking
- 🧭 **Progressive Recall** — Two-layer retrieval (`recall_index` → `recall_by_ids`) saves 5–10× tokens vs. full payload fetches
- 🛡️ **Privacy Filter** — `<private>`/`<secret>` tag stripping + automatic redaction of credential patterns (AWS/Stripe/GitHub/OpenAI/Anthropic/JWT/PEM) on `content`, `title`, `description`, `source_url` before persistence
- 🕸️ **Knowledge Graph** — Auto-extract entities and relations; query connectivity, paths, subgraphs
- 🤖 **RAG Answer** — `answer(question)` tool combines multi-query recall with LLM synthesis via Ollama
- 🧠 **Profile Generation** — Automatic user profile via local Ollama LLM
- 📁 **Project Support** — Organize memories by project (one Chroma collection per project)
- 🔌 **MCP Server** — Direct integration with Claude Desktop / Claude Code
- 📦 **Zero Config** — Works out of the box

## Installation

```bash
git clone https://github.com/granaria/local-supermemory.git
cd local-supermemory
pip install -e .
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for profile generation + RAG answers + higher-quality embeddings)

## Claude Desktop / Claude Code Configuration

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

Restart the Claude client.

## MCP Tools

### Memory core
| Tool | Description |
|------|-------------|
| `memory` | Save or forget information (`action`: save/forget). Privacy filter runs on content + metadata fields before persistence. |
| `recall` | Semantic search with optional profile summary. |
| `recall_multi` | Multi-query recall — Ollama paraphrases the query, deduplicates per memory_id with best score. |
| `answer` | RAG: retrieves context, synthesizes an answer via Ollama, returns answer + justification + sources. |
| `list_projects` | List all memory projects with counts. |
| `stats` | Storage statistics — total memories, chunks, embedding provider, knowledge graph stats. |
| `whoami` | Generate user profile from memories. |
| `rechunk` | Re-chunk all memories into the current schema (idempotent; useful after schema upgrades). |

### Progressive disclosure (token-efficient)
| Tool | Description |
|------|-------------|
| `recall_index` | **Layer 1** — returns a slim index (`memory_id, title, mem_type, created_at, score, project`). ~500 tokens vs ~8000 for full `recall`. |
| `recall_by_ids` | **Layer 2** — full Volltext only for specific memory_ids returned by `recall_index`. Uses SQLite as authoritative source for chunked memories. Max 20 IDs per call. |

### Knowledge graph
| Tool | Description |
|------|-------------|
| `graph_add_entity` | Create or update an entity (name, type, properties). |
| `graph_add_relation` | Create a relation between two entities. |
| `graph_link_memory` | Link a memory_id to an entity. |
| `graph_query` | Query the graph: `find_connected`, `shortest_path`, `subgraph`, `relations`, `search`, `entity_memories`. |
| `graph_stats` | Entities, relations, memory-links, type distributions. |

## Usage Examples

**Save a memory:**
```
Remember that my email is user@example.com
```

**Save with hard-private block (never persisted):**
```
Save this: API design doc finalized <private>auth token: sk_live_abc123</private> next review in a week
→ The <private>…</private> block is stripped; credential patterns are additionally redacted.
```

**Search memories (progressive):**
```
1. recall_index(query="trading setup", n_results=15)  → slim index
2. recall_by_ids(ids=["<top-2-ids>"])                  → full content
```

**Semantic search (classic):**
```
What do you know about my trading setup?
```

**Ask a question (RAG):**
```
answer(question="Which tools do I use for market research?", project="openclaw-trading")
→ synthesized answer + justification + source memories.
```

**Project-scoped memory:**
```
Save to project "work": Meeting notes from today...
```

**Graph exploration:**
```
graph_query(action="find_connected", entity="Dr2Jo Weigl", max_depth=2)
```

## Data Storage

All data is stored locally in `~/.granaria.supermemory/`:

```
~/.granaria.supermemory/
├── memories.db     # SQLite: metadata, projects, profile cache, full Volltext per memory_id
├── graph.db        # SQLite: entities, relations, memory links
└── chroma/         # ChromaDB: vector embeddings, one collection per project
```

Override the path via `MemoryStore(data_dir="…")` if you want to run multiple instances.

## Profile Generation

If [Ollama](https://ollama.ai) is running locally, the `whoami` tool and profile aggregation use it to generate an intelligent profile summary from your memories. Configure the model in `local_supermemory/profile.py`:

```python
OLLAMA_MODEL = "qwen2.5:32b"  # or any model you have
```

Without Ollama, a simple keyword-based fallback profile is generated.

## Embeddings

- **Ollama available** → embeddings via Ollama's embedding model (higher quality, consistent with your LLM)
- **Ollama not available** → ChromaDB's default embedding model (still works, no setup required)

## Privacy Model

The privacy filter runs before every `memory` save on **all** persisted fields (content, title, description, source_url):

- **Tier 1 — Hard strip:** `<private>…</private>` and `<secret>…</secret>` blocks removed (case-insensitive, multi-occurrence, multiline). Unclosed tags reject the save entirely to prevent accidental leaks.
- **Tier 2 — Pattern redact:** 9 credential shapes replaced with `[REDACTED:<type>]`:

| Pattern | Example |
|---|---|
| AWS access key | `AKIA…`, `ASIA…` |
| Stripe (live + test) | `sk_live_…`, `sk_test_…` |
| GitHub PAT/OAuth | `ghp_…`, `gho_…` |
| OpenAI (legacy) | `sk-…T3BlbkFJ…` |
| Anthropic | `sk-ant-…` |
| JWT | `eyJ…eyJ…` |
| PEM private key | `-----BEGIN … PRIVATE KEY-----` |

The response always reports what was filtered (`🔒 title: patterns redacted: stripe_live_key×1; content: 1 private block(s) stripped`) so you see immediately if something was stripped.

## Architecture

```
┌──────────────────────┐     ┌──────────────────┐
│ Claude Desktop/Code  │────▶│   MCP Server     │
└──────────────────────┘     └────────┬─────────┘
                                      │
                             ┌────────┴──────────┐
                             │ Privacy Filter    │  (phase1/hooks)
                             │ Progressive Recall│  (phase1/tools)
                             └────────┬──────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
             ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
             │  ChromaDB   │   │   SQLite    │   │   SQLite    │
             │  (vectors,  │   │ (memories,  │   │  (knowledge │
             │  per project)│  │  metadata)  │   │   graph)    │
             └─────────────┘   └─────────────┘   └─────────────┘
                    │
             ┌──────▼──────┐
             │   Ollama    │  (optional: embeddings + RAG + profile)
             └─────────────┘
```

## Dashboard

Optional local web UI for toggling capture and privacy behavior — runs as a
separate process, hot-reloaded by the MCP server on each save.

```bash
python -m phase1.dashboard            # → http://127.0.0.1:7333
# or, if installed as a script:
granaria-dashboard
```

The dashboard lets you:

- Toggle the privacy filter globally, per tier (tags vs patterns), per
  credential pattern (AWS, Stripe, GitHub, …)
- Set a max content length and a per-project blocklist (saves to blocked
  projects are rejected)
- Switch the knowledge-graph auto-extraction on/off as the default
- Inspect live stats (memories, chunks, projects, graph) and recent
  filter events

State is stored in `~/.granaria.supermemory/config.json`; the event log
lives at `~/.granaria.supermemory/dashboard_audit.jsonl`.

## Development

Run the Phase 1 unit tests:

```bash
python phase1/tests/test_phase1.py       # 34 tests: privacy filter, progressive recall, helpers
python phase1/tests/test_dashboard.py    # 22 tests: config, policy, filter toggles
```

## License

MIT License — see [LICENSE](LICENSE)

## Contributing

Contributions welcome — please open an issue or PR.

## Acknowledgments

Inspired by [Supermemory](https://supermemory.ai) and by [claude-mem](https://github.com/thedotmack/claude-mem)'s two-layer disclosure and dual-tag privacy patterns (AGPL-3.0; concepts reimplemented from scratch under MIT here).
