# Local Supermemory – Entwicklerhandbuch

Ein lokaler MCP-Server für semantische Memory-Speicherung mit ChromaDB und Ollama.

## Architektur

```
┌──────────────────────────────────────────────────────────────┐
│                      Claude Desktop                          │
└──────────────────────────┬───────────────────────────────────┘
                           │ MCP Protocol (JSON-RPC über stdio)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    server.py (MCP Server)                    │
│  - Registriert 5 Tools: memory, recall, list_projects,       │
│    stats, whoami                                             │
│  - Verwendet mcp.server.fastmcp für Tool-Dekoratoren         │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐
│    store.py      │              │   profile.py     │
│   MemoryStore    │              │  ProfileEngine   │
└────────┬─────────┘              └────────┬─────────┘
         │                                 │
    ┌────┴────┐                           │
    ▼         ▼                           ▼
┌───────┐ ┌───────┐                 ┌──────────┐
│ChromaDB│ │SQLite │                │  Ollama  │
│vectors │ │metadata│               │  (LLM)   │
└───────┘ └───────┘                 └──────────┘
```

## Datenfluss

**Speichern:**
```
User: "Merke dir: meine Email ist x@y.de"
  → server.py: memory(action="save", content="...")
    → store.save(content, project)
      → ChromaDB: Embedding generieren + speichern
      → SQLite: Metadaten (id, timestamp, project)
```

**Suchen:**
```
User: "Was weißt du über meine Email?"
  → server.py: recall(query="email")
    → store.search(query, n=5)
      → ChromaDB: Cosine-Similarity auf Embeddings
      → SQLite: Metadaten laden
    → profile.generate(memories) [optional]
      → Ollama: LLM-Zusammenfassung
```

## Dateistruktur

```
local_supermemory/
├── __init__.py      # Package-Definition
├── server.py        # MCP-Server, Tool-Registrierung
├── store.py         # Datenbank-Logik (ChromaDB + SQLite)
└── profile.py       # LLM-basierte Profil-Generierung
```

---

## store.py – MemoryStore

Der Kern der Datenhaltung. Dual-Storage-Ansatz:

### Warum zwei Datenbanken?

| ChromaDB | SQLite |
|----------|--------|
| Vektor-Embeddings für semantische Suche | Strukturierte Metadaten |
| Schnelle Similarity-Search | ACID-Transaktionen |
| Keine JOINs möglich | Flexible Abfragen |

### Initialisierung

```python
class MemoryStore:
    def __init__(self, data_dir: str = None):
        # Default: ~/.local-supermemory/
        self.data_dir = Path(data_dir or Path.home() / ".local-supermemory")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB: Persistenter Client (nicht in-memory!)
        self.chroma = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma")
        )
        # Collection = "Tabelle" in ChromaDB
        # cosine = Winkel zwischen Vektoren (0-1, höher = ähnlicher)
        self.collection = self.chroma.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}  # HNSW = Approximate NN
        )
        
        # SQLite für Metadaten
        self.db = sqlite3.connect(str(self.data_dir / "memories.db"))
        self._init_db()
```

**HNSW (Hierarchical Navigable Small World):** Algorithmus für approximative Nearest-Neighbor-Suche. Schneller als Brute-Force, minimal ungenauer.

### Speichern

```python
def save(self, content: str, project: str = "default") -> str:
    memory_id = str(uuid.uuid4())
    
    # ChromaDB erwartet Listen, keine Einzelwerte
    self.collection.add(
        ids=[memory_id],
        documents=[content],
        # Wichtig: ChromaDB crasht bei leeren Metadaten!
        metadatas=[{"project": project or "default"}]
    )
    
    # SQLite: Zusätzliche Metadaten
    self.db.execute(
        "INSERT INTO memories (id, content, project, created_at) VALUES (?,?,?,?)",
        (memory_id, content, project, datetime.now().isoformat())
    )
    self.db.commit()
    return memory_id
```

### Suchen

```python
def search(self, query: str, n: int = 5, project: str = None) -> list:
    # ChromaDB macht automatisch:
    # 1. Query → Embedding (gleicher Encoder wie beim Speichern)
    # 2. Cosine-Similarity gegen alle gespeicherten Vektoren
    # 3. Top-N zurückgeben
    
    where = {"project": project} if project else None
    
    results = self.collection.query(
        query_texts=[query],  # Nicht query_text (Singular)!
        n_results=n,
        where=where,
        include=["documents", "distances", "metadatas"]
    )
    
    # Ergebnis-Format umwandeln
    memories = []
    for i, doc in enumerate(results["documents"][0]):
        memories.append({
            "content": doc,
            "score": 1 - results["distances"][0][i],  # distance → similarity
            "project": results["metadatas"][0][i].get("project", "default")
        })
    return memories
```

**Warum `1 - distance`?** ChromaDB gibt Distanz zurück (0 = identisch), wir wollen Similarity (1 = identisch).

### Löschen

```python
def forget(self, content: str) -> bool:
    # Problem: Wir haben nur den Content, nicht die ID
    # Lösung: Suchen, dann ID extrahieren
    
    results = self.collection.query(
        query_texts=[content],
        n_results=1,
        include=["documents"]
    )
    
    if not results["ids"][0]:
        return False
    
    memory_id = results["ids"][0][0]
    self.collection.delete(ids=[memory_id])
    self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    self.db.commit()
    return True
```

---

## profile.py – ProfileEngine

Generiert ein Benutzerprofil aus allen Memories mittels lokalem LLM.

### Ollama-Integration

```python
class ProfileEngine:
    def __init__(self, model: str = "qwen2.5:32b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def generate(self, memories: list[str]) -> str:
        # Alle Memories als Kontext
        context = "\n".join(f"- {m}" for m in memories)
        
        prompt = f"""Analyze these memories and create a brief user profile.
Focus on: interests, skills, projects, preferences.

Memories:
{context}

Profile (3-5 sentences):"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False  # Wichtig: Ganze Antwort auf einmal
                },
                timeout=60
            )
            return response.json()["response"]
        except:
            # Fallback: Einfache Keyword-Extraktion
            return self._fallback_profile(memories)
```

### Fallback ohne LLM

```python
def _fallback_profile(self, memories: list[str]) -> str:
    # Häufigste Wörter extrahieren (stopwords ignorieren)
    words = []
    stopwords = {"the", "a", "is", "are", "and", "or", "to", "for", ...}
    
    for m in memories:
        words.extend([
            w.lower() for w in m.split() 
            if w.lower() not in stopwords and len(w) > 3
        ])
    
    # Counter für Häufigkeit
    from collections import Counter
    common = Counter(words).most_common(10)
    
    return f"Keywords: {', '.join(w for w, _ in common)}"
```

---

## server.py – MCP Server

Verbindet alles mit Claude Desktop über das Model Context Protocol.

### FastMCP Setup

```python
from mcp.server.fastmcp import FastMCP

# Name erscheint in Claude Desktop
mcp = FastMCP("local-supermemory")

store = MemoryStore()
profile_engine = ProfileEngine()
```

### Tool-Definitionen

```python
@mcp.tool()
def memory(content: str, action: str = "save", project: str = None) -> str:
    """
    Save or forget information.
    
    Args:
        content: The memory content
        action: "save" or "forget"
        project: Optional project scope
    """
    if action == "forget":
        success = store.forget(content)
        return "Forgotten." if success else "Not found."
    
    memory_id = store.save(content, project)
    return f"Saved: {memory_id}"


@mcp.tool()
def recall(query: str, n: int = 5, project: str = None, 
           include_profile: bool = True) -> str:
    """
    Search memories semantically.
    """
    memories = store.search(query, n, project)
    
    if not memories:
        return "No memories found."
    
    # Formatierte Ausgabe
    output = []
    for m in memories:
        score_pct = int(m["score"] * 100)
        output.append(f"[{score_pct}%] {m['content']}")
    
    result = "\n".join(output)
    
    # Optional: Profil anhängen
    if include_profile:
        all_memories = store.get_all()
        profile = profile_engine.generate([m["content"] for m in all_memories])
        result += f"\n\n---\nProfile: {profile}"
    
    return result


@mcp.tool()
def stats() -> str:
    """Storage statistics."""
    count = store.collection.count()
    projects = store.get_projects()
    return f"Memories: {count}\nProjects: {', '.join(projects)}"
```

### Server starten

```python
if __name__ == "__main__":
    # stdio = Kommunikation über stdin/stdout (nicht HTTP!)
    mcp.run(transport="stdio")
```

---

## SQLite Schema

```sql
-- Haupttabelle
CREATE TABLE memories (
    id TEXT PRIMARY KEY,           -- UUID
    content TEXT NOT NULL,         -- Memory-Text
    project TEXT DEFAULT 'default',-- Projekt-Zuordnung
    created_at TEXT NOT NULL       -- ISO-Timestamp
);

-- Projekt-Verwaltung (optional)
CREATE TABLE projects (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

-- Profil-Cache (vermeidet wiederholte LLM-Calls)
CREATE TABLE profile_cache (
    id INTEGER PRIMARY KEY,
    profile TEXT,
    updated_at TEXT
);

-- Indizes für Performance
CREATE INDEX idx_memories_project ON memories(project);
CREATE INDEX idx_memories_created ON memories(created_at);
```

---

## ChromaDB Internals

### Embedding-Modell

ChromaDB verwendet standardmäßig `all-MiniLM-L6-v2`:
- 384-dimensionale Vektoren
- Multilingual (auch Deutsch)
- ~80MB Modellgröße

Eigenes Modell nutzen:
```python
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = chroma.get_or_create_collection(
    name="memories",
    embedding_function=ef
)
```

### Persistenz

```
~/.local-supermemory/chroma/
├── chroma.sqlite3          # Metadaten + IDs
└── <uuid>/                 # Collection-Daten
    ├── data_level0.bin     # HNSW-Graph
    ├── header.bin          # Index-Header
    ├── index_metadata.json # Konfiguration
    └── length.bin          # Vektor-Längen
```

---

## Erweiterungen

### Eigenes Embedding-Modell

```python
# In store.py
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings"
)

self.collection = self.chroma.get_or_create_collection(
    name="memories",
    embedding_function=ef
)
```

### Memory-Tags

```python
def save(self, content: str, project: str = None, tags: list = None):
    metadata = {
        "project": project or "default",
        "tags": ",".join(tags) if tags else ""
    }
    self.collection.add(
        ids=[memory_id],
        documents=[content],
        metadatas=[metadata]
    )
```

### Export/Import

```python
def export_json(self) -> str:
    memories = self.get_all()
    return json.dumps(memories, indent=2, ensure_ascii=False)

def import_json(self, json_str: str):
    memories = json.loads(json_str)
    for m in memories:
        self.save(m["content"], m.get("project"))
```

### Web-API (FastAPI)

```python
from fastapi import FastAPI
app = FastAPI()
store = MemoryStore()

@app.post("/memory")
def save_memory(content: str, project: str = None):
    return {"id": store.save(content, project)}

@app.get("/search")
def search(q: str, n: int = 5):
    return store.search(q, n)
```

---

## Debugging

### ChromaDB direkt abfragen

```python
import chromadb
client = chromadb.PersistentClient(path="~/.local-supermemory/chroma")
collection = client.get_collection("memories")

# Alle Einträge
print(collection.peek(10))

# Count
print(collection.count())
```

### MCP-Server testen

```bash
# Manuell starten
python -m local_supermemory.server

# Dann JSON-RPC über stdin:
{"jsonrpc": "2.0", "method": "tools/list", "id": 1}
```

### Logs aktivieren

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Performance

| Operation | ~Zeit | Notizen |
|-----------|-------|---------|
| save() | 50-100ms | Embedding-Berechnung dominiert |
| search() | 10-50ms | HNSW ist sehr schnell |
| profile (Ollama) | 2-10s | Abhängig vom Modell |
| profile (Fallback) | <10ms | Nur String-Ops |

### Optimierungen

1. **Batch-Insert:** `collection.add(ids=[...], documents=[...])`
2. **Profil-Caching:** Nur neu generieren wenn Memories sich ändern
3. **Kleineres Embedding-Modell:** Trade-off Qualität vs. Speed

---

## Lizenz

MIT – Nutzung, Modifikation und Weitergabe erlaubt.
