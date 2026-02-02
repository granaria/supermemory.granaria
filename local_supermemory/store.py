"""Memory Store - ChromaDB + SQLite Backend"""
import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings


class MemoryStore:
    def __init__(self, data_dir: str = "~/.local-supermemory"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.db_path = self.data_dir / "memories.db"
        self._init_db()
    
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
        return self.chroma.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    
    def _gen_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def save(self, content: str, project: str = "default", metadata: Optional[dict] = None) -> dict:
        mid = self._gen_id(content)
        now = datetime.now().isoformat()
        coll = self._get_collection(project)
        meta = metadata if metadata else {"project": project}
        
        existing = coll.get(ids=[mid])
        if existing and existing['ids']:
            coll.update(ids=[mid], documents=[content], metadatas=[meta])
            action = "updated"
        else:
            coll.add(ids=[mid], documents=[content], metadatas=[meta])
            action = "created"
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO memories (id, content, project, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at
        """, (mid, content, project, now, now, json.dumps(meta)))
        conn.execute("INSERT OR IGNORE INTO projects (name, description, created_at) VALUES (?, ?, ?)",
                     (project, f"Projekt {project}", now))
        conn.execute("DELETE FROM profile_cache WHERE project = ?", (project,))
        conn.commit()
        conn.close()
        return {"id": mid, "action": action, "project": project}
    
    def forget(self, content: str, project: str = "default") -> dict:
        mid = self._gen_id(content)
        try:
            self._get_collection(project).delete(ids=[mid])
        except: pass
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("DELETE FROM memories WHERE id = ? AND project = ?", (mid, project))
        deleted = cur.rowcount > 0
        conn.execute("DELETE FROM profile_cache WHERE project = ?", (project,))
        conn.commit()
        conn.close()
        return {"id": mid, "deleted": deleted}
    
    def recall(self, query: str, project: str = "default", n: int = 15) -> list:
        coll = self._get_collection(project)
        if coll.count() == 0:
            return []
        res = coll.query(query_texts=[query], n_results=min(n, coll.count()))
        memories = []
        if res and res['documents'] and res['documents'][0]:
            for i, doc in enumerate(res['documents'][0]):
                dist = res['distances'][0][i] if res['distances'] else 0
                memories.append({
                    "content": doc,
                    "similarity": round(max(0, 1 - dist) * 100),
                    "id": res['ids'][0][i] if res['ids'] else None
                })
        return memories
    
    def get_all(self, project: str = "default") -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM memories WHERE project = ? ORDER BY updated_at DESC", (project,))
        mems = [dict(r) for r in cur.fetchall()]
        conn.close()
        return mems
    
    def list_projects(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        projects = []
        for row in conn.execute("SELECT * FROM projects ORDER BY name"):
            p = dict(row)
            p['count'] = conn.execute("SELECT COUNT(*) FROM memories WHERE project=?", (p['name'],)).fetchone()[0]
            projects.append(p)
        conn.close()
        return projects
    
    def get_profile(self, project: str = "default") -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT profile FROM profile_cache WHERE project=?", (project,)).fetchone()
        conn.close()
        return row[0] if row else None
    
    def set_profile(self, project: str, profile: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO profile_cache (project, profile, generated_at) VALUES (?, ?, datetime('now'))",
                     (project, profile))
        conn.commit()
        conn.close()
    
    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        by_proj = {r[0]: r[1] for r in conn.execute("SELECT project, COUNT(*) FROM memories GROUP BY project")}
        conn.close()
        return {"total": total, "projects": projects, "by_project": by_proj, "path": str(self.data_dir)}
