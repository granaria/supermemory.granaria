"""Memory Store v2.1 — ChromaDB + Ollama Embeddings + Knowledge Graph
+ Sentence-Chunking + Score-Normalisierung + Metadata-Erweiterung.

Phase 1 Änderungen (portiert aus supermemoryai/supermemory v2):
  * save():   content → pysbd-Sätze → Chunks mit Overlap → N ChromaDB-Einträge
              pro Memory. SQLite bleibt Source of Truth für den Volltext.
  * recall(): Chunks werden abgefragt, nach parent memory_id dedupt, der
              beste Match pro Memory wird zurückgegeben. Zusätzlich zur
              absoluten `similarity` kommt ein relatives `normalised_score`
              (1..100, min-max über das aktuelle Query-Resultset).
  * save():   optionale Metadaten title / source_url / description werden
              in ChromaDB-metadata und SQLite-metadata JSON persistiert.

Rückwärtskompatibilität: Alte ChromaDB-Einträge (ohne memory_id, chunk_index
in metadata) werden in recall() als Legacy-Single-Chunk behandelt. Neues
Tool `rechunk_all()` für optionale Migration.
"""
import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

from .embeddings import get_ollama_ef
from .graph import KnowledgeGraph
from .chunking import chunk_text


# ChromaDB ID-Konvention für Chunks: {memory_id}#c{index:03d}
# - "#" kommt in sha256-hex nie vor → kollisionsfrei zu Legacy-IDs
# - 3-stelliger Index reicht bis 999 Chunks pro Memory (≈ 800 KB Content)
CHUNK_ID_SEPARATOR = "#c"


class MemoryStore:
    def __init__(self, data_dir: str = "~/.granaria.supermemory"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._ollama_ef = get_ollama_ef()
        self._use_ollama = self._ollama_ef.is_available()

        self.chroma = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

        self.db_path = self.data_dir / "memories.db"
        self._init_db()

        self.graph = KnowledgeGraph(self.db_path)

        if self._use_ollama:
            print(f"[Store] Ollama Embeddings aktiv ({self._ollama_ef.model})")
        else:
            print("[Store] ChromaDB Default Embeddings (Ollama nicht verfügbar)")

    @property
    def embedding_info(self) -> str:
        if self._use_ollama:
            return f"Ollama ({self._ollama_ef.model})"
        return "ChromaDB Default"

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                project TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS projects (
                name TEXT PRIMARY KEY,
                description TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS profile_cache (
                project TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
            INSERT OR IGNORE INTO projects (name, description, created_at)
            VALUES ('default', 'Standard-Projekt', datetime('now'));
        """)
        conn.commit()
        conn.close()

    def _get_collection(self, project: str = "default"):
        name = f"memories_{project.replace('-', '_')}"
        kwargs = {"name": name, "metadata": {"hnsw:space": "cosine"}}
        if self._use_ollama:
            kwargs["embedding_function"] = self._ollama_ef
        return self.chroma.get_or_create_collection(**kwargs)

    def _gen_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ═══════════════════════════════════════════════════════════════
    # SAVE — mit Sentence-Chunking + Metadata-Erweiterung
    # ═══════════════════════════════════════════════════════════════

    def save(self, content: str, project: str = "default",
             title: Optional[str] = None,
             source_url: Optional[str] = None,
             description: Optional[str] = None,
             language: str = "auto",
             metadata: Optional[dict] = None,
             auto_extract: bool = True) -> dict:
        """Speichere Memory als N Chunks + optional Graph-Extraktion.

        Args:
            content: Der Memory-Volltext (SQLite).
            project: Projekt-Namespace.
            title: Optional, Titel (z.B. Seiten-Titel bei Web-Clippings).
            source_url: Optional, Herkunfts-URL.
            description: Optional, Kurzbeschreibung/Meta-Description.
            language: Chunking-Sprache ("de"/"en"/"auto" — default auto).
            metadata: Zusätzliche freie Metadaten (mergt mit den Feldern oben).
            auto_extract: LLM-Entity-Extraktion für Knowledge Graph.
        """
        mid = self._gen_id(content)
        now = datetime.now().isoformat()
        coll = self._get_collection(project)

        # Chunking
        chunks = chunk_text(content, language=language)
        n_chunks = len(chunks)
        if n_chunks == 0:
            return {"error": "empty content"}

        # Base-Metadata für ChromaDB (muss flat + primitive sein)
        base_meta = {
            "project": project,
            "memory_id": mid,
            "chunk_count": n_chunks,
            "title": title or "",
            "source_url": source_url or "",
            "description": description or "",
        }

        # Idempotenz: existierende Chunks dieser Memory entfernen
        # (sowohl neue Chunk-Einträge als auch potentieller Legacy-Single-Entry
        # unter der blanken memory_id)
        try:
            existing = coll.get(where={"memory_id": mid})
            if existing and existing.get("ids"):
                coll.delete(ids=existing["ids"])
        except Exception:
            pass
        try:
            legacy = coll.get(ids=[mid])
            if legacy and legacy.get("ids"):
                coll.delete(ids=[mid])
        except Exception:
            pass

        # Neue Chunks einfügen
        ids = [f"{mid}{CHUNK_ID_SEPARATOR}{i:03d}" for i in range(n_chunks)]
        metas = [{**base_meta, "chunk_index": i} for i in range(n_chunks)]
        coll.add(ids=ids, documents=chunks, metadatas=metas)

        # SQLite: kombinierte metadata-JSON (Volltext bleibt in `content`)
        sql_meta = {"project": project, "chunk_count": n_chunks}
        if title:
            sql_meta["title"] = title
        if source_url:
            sql_meta["source_url"] = source_url
        if description:
            sql_meta["description"] = description
        if metadata:
            sql_meta.update(metadata)

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT id FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        action = "updated" if cur else "created"

        conn.execute("""
            INSERT INTO memories (id, content, project, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
        """, (mid, content, project, now, now, json.dumps(sql_meta)))
        conn.execute(
            "INSERT OR IGNORE INTO projects (name, description, created_at) VALUES (?, ?, ?)",
            (project, f"Projekt {project}", now)
        )
        conn.execute("DELETE FROM profile_cache WHERE project = ?", (project,))
        conn.commit()
        conn.close()

        # Knowledge-Graph Auto-Extract auf dem Gesamttext
        graph_result = None
        if auto_extract and len(content) > 20:
            graph_result = self.graph.extract_and_link(mid, content, project)

        result = {
            "id": mid,
            "action": action,
            "project": project,
            "chunks": n_chunks,
        }
        if graph_result:
            result["graph"] = graph_result
        return result

    def forget(self, content: str, project: str = "default") -> dict:
        mid = self._gen_id(content)
        coll = self._get_collection(project)

        # Alle Chunk-Einträge zu dieser Memory löschen (neue Form)
        deleted_chunks = 0
        try:
            existing = coll.get(where={"memory_id": mid})
            if existing and existing.get("ids"):
                deleted_chunks = len(existing["ids"])
                coll.delete(ids=existing["ids"])
        except Exception:
            pass
        # Legacy-Eintrag unter blanker mid (falls vorhanden)
        try:
            coll.delete(ids=[mid])
        except Exception:
            pass

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("DELETE FROM memories WHERE id = ? AND project = ?",
                           (mid, project))
        deleted = cur.rowcount > 0
        conn.execute("DELETE FROM profile_cache WHERE project = ?", (project,))
        conn.commit()
        conn.close()
        return {"id": mid, "deleted": deleted, "chunks_removed": deleted_chunks}

    # ═══════════════════════════════════════════════════════════════
    # RECALL — mit Chunk-Dedup + Min-Max Score-Normalisierung
    # ═══════════════════════════════════════════════════════════════

    def recall(self, query: str, project: str = "default", n: int = 15) -> list:
        """Semantische Suche mit Chunk-Dedup und Score-Normalisierung.

        Returnt pro gefundener Memory:
            - Volltext aus SQLite (content)
            - matched_chunk: der am besten matchende Chunk (nur bei chunk_count > 1)
            - similarity: absolute cosine-similarity (0..100)
            - normalised_score: min-max relativ zum Resultset (1..100)
            - title, source_url, description aus Metadata
        """
        coll = self._get_collection(project)
        coll_size = coll.count()
        if coll_size == 0:
            return []

        # Wir fragen das 3-fache an Chunks ab, weil nach memory_id-Dedup
        # weniger übrig bleibt. Upper bound = coll_size.
        n_chunks_to_fetch = min(max(n * 3, n), coll_size)

        res = coll.query(query_texts=[query], n_results=n_chunks_to_fetch)

        # Rohdaten extrahieren
        raw_hits: list[dict] = []
        if res and res.get("documents") and res["documents"][0]:
            docs = res["documents"][0]
            dists = res["distances"][0] if res.get("distances") else [0] * len(docs)
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            ids = res["ids"][0] if res.get("ids") else [None] * len(docs)

            for doc, dist, md, raw_id in zip(docs, dists, metas, ids):
                md = md or {}
                # Legacy-Fallback: alte Einträge haben keine memory_id in metadata
                # → raw_id selbst ist die memory_id (sha256[:16], ohne #c-Suffix)
                mem_id = md.get("memory_id")
                if not mem_id:
                    mem_id = (raw_id.split(CHUNK_ID_SEPARATOR)[0]
                              if raw_id and CHUNK_ID_SEPARATOR in raw_id
                              else raw_id)
                raw_hits.append({
                    "memory_id": mem_id,
                    "chunk_text": doc,
                    "distance": dist,
                    "similarity": max(0.0, 1.0 - dist) * 100,
                    "chunk_index": md.get("chunk_index", 0),
                    "chunk_count": md.get("chunk_count", 1),
                    "title": md.get("title", ""),
                    "source_url": md.get("source_url", ""),
                    "description": md.get("description", ""),
                })

        # Dedup: pro memory_id nur den besten Chunk behalten
        best_per_memory: dict[str, dict] = {}
        for hit in raw_hits:
            mid = hit["memory_id"]
            if mid not in best_per_memory or hit["similarity"] > best_per_memory[mid]["similarity"]:
                best_per_memory[mid] = hit

        # Nach Similarity sortieren, auf n begrenzen
        deduped = sorted(best_per_memory.values(),
                         key=lambda x: -x["similarity"])[:n]

        # v2-Port: Min-Max Score-Normalisierung
        # (1..100, identische Scores → 50)
        if deduped:
            sims = [h["similarity"] for h in deduped]
            lo, hi = min(sims), max(sims)
            for h in deduped:
                if hi == lo:
                    h["normalised_score"] = 50
                else:
                    h["normalised_score"] = round(
                        ((h["similarity"] - lo) / (hi - lo)) * 99 + 1
                    )

        # Output: Volltext aus SQLite ziehen, Chunk nur als Info mitgeben
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        result: list[dict] = []
        for h in deduped:
            row = conn.execute(
                "SELECT content FROM memories WHERE id = ?", (h["memory_id"],)
            ).fetchone()
            full_content = row["content"] if row else h["chunk_text"]
            entry = {
                "id": h["memory_id"],
                "content": full_content,
                "similarity": round(h["similarity"]),
                "normalised_score": h["normalised_score"],
                "chunk_count": h["chunk_count"],
            }
            # Matched Chunk nur zeigen, wenn er sich vom Volltext unterscheidet
            if h["chunk_count"] > 1:
                entry["matched_chunk"] = h["chunk_text"]
                entry["matched_chunk_index"] = h["chunk_index"]
            # Metadata nur zeigen wenn nicht leer
            if h["title"]:
                entry["title"] = h["title"]
            if h["source_url"]:
                entry["source_url"] = h["source_url"]
            if h["description"]:
                entry["description"] = h["description"]
            result.append(entry)
        conn.close()

        return result

    # ═══════════════════════════════════════════════════════════════
    # Rest unverändert (bis auf rechunk_all + stats um chunk_count erweitert)
    # ═══════════════════════════════════════════════════════════════

    def get_all(self, project: str = "default") -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM memories WHERE project = ? ORDER BY updated_at DESC",
            (project,)
        )
        mems = [dict(r) for r in cur.fetchall()]
        conn.close()
        return mems

    def get_memory_by_id(self, memory_id: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_projects(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        projects = []
        for row in conn.execute("SELECT * FROM projects ORDER BY name"):
            p = dict(row)
            p["count"] = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project=?", (p["name"],)
            ).fetchone()[0]
            projects.append(p)
        conn.close()
        return projects

    def get_profile(self, project: str = "default") -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT profile FROM profile_cache WHERE project=?", (project,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def set_profile(self, project: str, profile: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO profile_cache (project, profile, generated_at) "
            "VALUES (?, ?, datetime('now'))",
            (project, profile)
        )
        conn.commit()
        conn.close()

    def rebuild_graph(self, project: str = None) -> dict:
        """Rebuild Knowledge Graph aus allen bestehenden Memories."""
        if project:
            memories = self.get_all(project)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            memories = [dict(r) for r in conn.execute(
                "SELECT * FROM memories ORDER BY updated_at"
            ).fetchall()]
            conn.close()

        total_entities = 0
        total_relations = 0
        processed = 0
        errors = 0

        for mem in memories:
            result = self.graph.extract_and_link(
                mem["id"], mem["content"],
                mem.get("project", "default")
            )
            if result.get("error"):
                errors += 1
            else:
                total_entities += result.get("entities", 0)
                total_relations += result.get("relations", 0)
            processed += 1

        return {
            "processed": processed,
            "entities_extracted": total_entities,
            "relations_extracted": total_relations,
            "errors": errors
        }

    def rechunk_all(self, project: str = None) -> dict:
        """Alle Memories neu chunken + neu embedden mit aktuellem Provider.

        Zweck: Legacy-Single-Entries (ohne chunk_index/memory_id metadata)
        ins neue Chunk-Schema bringen. Funktioniert mit jedem Embedding-
        Provider (Ollama wenn verfügbar, sonst ChromaDB-Default).

        Strategie: pro Projekt die Collection droppen und mit save()
        für jede Memory neu aufbauen. save() übernimmt Chunking,
        Metadata und Idempotenz. Graph-Extraktion wird übersprungen
        (separates rebuild_graph() verfügbar).
        """
        if project:
            projects = [project]
        else:
            projects = [p["name"] for p in self.list_projects()]

        total = 0
        errors = 0
        for proj in projects:
            memories = self.get_all(proj)
            if not memories:
                continue

            # Collection droppen — die wird von save() neu angelegt
            name = f"memories_{proj.replace('-', '_')}"
            try:
                self.chroma.delete_collection(name)
            except Exception:
                pass

            for mem in memories:
                try:
                    meta = {}
                    if mem.get("metadata"):
                        try:
                            meta = json.loads(mem["metadata"])
                        except Exception:
                            pass
                    self.save(
                        mem["content"], proj,
                        title=meta.get("title"),
                        source_url=meta.get("source_url"),
                        description=meta.get("description"),
                        auto_extract=False,
                    )
                    total += 1
                except Exception:
                    errors += 1

        result = {
            "migrated": total,
            "projects": len(projects),
            "provider": self.embedding_info,
        }
        if errors:
            result["errors"] = errors
        return result

    def migrate_embeddings(self, project: str = None) -> dict:
        """Alias auf rechunk_all — gleicher Effekt.

        Historisch separates Konzept (Embedding-Provider-Wechsel), aber
        technisch identisch zur Chunk-Migration. Alias bleibt für
        Rückwärtskompatibilität von Python-API-Callern.
        """
        return self.rechunk_all(project)

    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        by_proj = {
            r[0]: r[1]
            for r in conn.execute("SELECT project, COUNT(*) FROM memories GROUP BY project")
        }
        conn.close()

        # Chunk-Statistiken aus ChromaDB
        total_chunks = 0
        for p in by_proj:
            try:
                coll = self._get_collection(p)
                total_chunks += coll.count()
            except Exception:
                pass

        graph_stats = self.graph.stats()

        return {
            "total_memories": total,
            "total_chunks": total_chunks,
            "chunks_per_memory": round(total_chunks / total, 2) if total else 0,
            "projects": projects,
            "by_project": by_proj,
            "path": str(self.data_dir),
            "embedding_provider": self.embedding_info,
            "graph": graph_stats
        }
