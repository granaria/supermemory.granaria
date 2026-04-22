"""Local Supermemory MCP Server v2 — mit Knowledge Graph + Ollama Embeddings"""
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .store import MemoryStore
from .profile import get_engine

# ── Phase 1: Privacy filter + progressive recall ────────────────
from phase1.hooks.privacy_filter import filter_content
from phase1.tools.recall_progressive import build_index, fetch_by_ids

server = Server("local-supermemory")
store = MemoryStore()


# ═══════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── Memory Tools (v1) ────────────────────────────────
        Tool(
            name="memory",
            description=(
                "Speichere oder vergesse Informationen. "
                "action: 'save'|'forget', content: Text, project: optional. "
                "Optionale Metadaten (nur bei save): title, source_url, description, language."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["save", "forget"], "default": "save"},
                    "content": {"type": "string", "maxLength": 200000},
                    "project": {"type": "string", "maxLength": 128},
                    "title": {"type": "string", "maxLength": 256,
                              "description": "Optional: Titel (z.B. bei Web-Clipping)"},
                    "source_url": {"type": "string", "maxLength": 2048,
                                   "description": "Optional: Herkunfts-URL"},
                    "description": {"type": "string", "maxLength": 1024,
                                    "description": "Optional: Kurzbeschreibung"},
                    "language": {"type": "string", "enum": ["de", "en", "auto"],
                                 "default": "auto",
                                 "description": "Sprache für Sentence-Chunking"}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="recall",
            description="Semantische Suche in Memories mit Profil-Aggregation",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 1000},
                    "project": {"type": "string", "maxLength": 128},
                    "include_profile": {"type": "boolean", "default": True},
                    "n_results": {"type": "integer", "default": 15}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_projects",
            description="Liste alle Projekte",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="stats",
            description="Speicher-Statistiken",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="whoami",
            description="Zeigt Benutzerinfo basierend auf Memories",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="rechunk",
            description=(
                "Migriere alle bestehenden Memories ins neue Chunk-Schema "
                "(Sentence-Chunking + Metadata). Sicher, idempotent, auch "
                "Legacy-Einträge werden neu gechunkt. project: optional "
                "(ohne → alle Projekte). Kann bei vielen Memories einige "
                "Sekunden dauern, weil jede neu embedded wird."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "maxLength": 128}
                }
            }
        ),
        Tool(
            name="recall_multi",
            description=(
                "Multi-Query Recall: Ollama paraphrasiert die Anfrage in "
                "mehrere Varianten, recallt alle parallel, dedupliziert "
                "pro Memory-ID mit bestem Score. Robuster bei Synonymen "
                "und mehrdeutigen Fragen als normales recall, aber langsamer "
                "(Query-Expansion braucht 2–10s)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 1000},
                    "project": {"type": "string", "maxLength": 128},
                    "n_results": {"type": "integer", "default": 15},
                    "expand_n": {
                        "type": "integer", "default": 3, "minimum": 0,
                        "maximum": 8,
                        "description": "Anzahl Paraphrasen (0 = wie recall)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="answer",
            description=(
                "Beantwortet eine Frage auf Basis der Memories (RAG). "
                "Kombiniert Multi-Query Recall, v2-Context-Score-Prompt "
                "und Ollama. Liefert Begründung (justification) + Antwort "
                "+ Quellenliste. Bei Ollama-Offline: Fallback auf reines "
                "Retrieval ohne LLM-Synthese."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "maxLength": 1000},
                    "project": {"type": "string", "maxLength": 128},
                    "n_context": {"type": "integer", "default": 8,
                                  "minimum": 1, "maximum": 20},
                    "use_multi_query": {"type": "boolean", "default": True}
                },
                "required": ["question"]
            }
        ),

        # ── Progressive Disclosure (Phase 1) ─────────────────
        Tool(
            name="recall_index",
            description=(
                "Progressive Recall Layer 1: liefert einen schlanken Index "
                "passender Memories (memory_id, title, mem_type, created_at, "
                "score, project) — ~500 Tokens statt ~8000. "
                "Danach gezielt recall_by_ids(ids=[...]) aufrufen, nur für "
                "die Memories, deren Volltext du wirklich brauchst."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 1000},
                    "n_results": {"type": "integer", "default": 15,
                                  "minimum": 1, "maximum": 50},
                    "project": {"type": "string", "maxLength": 128,
                                "description": "Pflicht wenn Memories in "
                                               "einem bestimmten Projekt "
                                               "liegen; default='default'"},
                    "mem_type": {"type": "string",
                                 "description": "Optional: Typ-Filter"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="recall_by_ids",
            description=(
                "Progressive Recall Layer 2: lädt den Volltext für "
                "spezifische memory_ids (aus recall_index). Max 20 IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20
                    },
                    "project": {"type": "string", "maxLength": 128,
                                "description": "default='default'"}
                },
                "required": ["ids"]
            }
        ),

        # ── Knowledge Graph Tools (v2) ───────────────────────
        Tool(
            name="graph_add_entity",
            description="Erstelle oder aktualisiere eine Entität im Knowledge Graph. name: Name, type: z.B. person/project/tool/company/concept, properties: optionales dict",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "default": "unknown"},
                    "properties": {"type": "object", "default": {}}
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="graph_add_relation",
            description="Erstelle eine Relation zwischen zwei Entitäten. source/target: Namen, relation_type: z.B. 'arbeitet_an', 'nutzt', 'kennt', 'gehört_zu'",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "source_type": {"type": "string", "default": "unknown"},
                    "target_type": {"type": "string", "default": "unknown"},
                    "properties": {"type": "object", "default": {}}
                },
                "required": ["source", "target", "relation_type"]
            }
        ),
        Tool(
            name="graph_link_memory",
            description="Verknüpfe eine Memory (per ID) mit einer Entität",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "entity_name": {"type": "string"},
                    "entity_type": {"type": "string", "default": "unknown"},
                    "project": {"type": "string", "default": "default"}
                },
                "required": ["memory_id", "entity_name"]
            }
        ),
        Tool(
            name="graph_query",
            description="Abfrage des Knowledge Graphs. action: 'find_connected'|'shortest_path'|'subgraph'|'relations'|'search'|'entity_memories'",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["find_connected", "shortest_path", "subgraph",
                                 "relations", "search", "entity_memories"]
                    },
                    "entity": {"type": "string", "description": "Entitäts-Name für find_connected, relations, entity_memories"},
                    "from_entity": {"type": "string", "description": "Start für shortest_path"},
                    "to_entity": {"type": "string", "description": "Ziel für shortest_path"},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Liste für subgraph"},
                    "query": {"type": "string", "description": "Suchbegriff für search"},
                    "entity_type": {"type": "string", "description": "Typ-Filter"},
                    "relation_type": {"type": "string", "description": "Relations-Typ-Filter"},
                    "max_depth": {"type": "integer", "default": 2}
                },
                "required": ["action"]
            }
        ),
        Tool(
            name="graph_stats",
            description="Knowledge Graph Statistiken: Entitäten, Relationen, Typen",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


# ═══════════════════════════════════════════════════════════════
# Tool Handlers
# ═══════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:

    # ── Memory Tools ─────────────────────────────────────────
    if name == "memory":
        action = args.get("action", "save")
        content = args.get("content", "")
        project = args.get("project", "default")
        if not content:
            return [TextContent(type="text", text="Fehler: content erforderlich")]
        if action == "save":
            # Phase 1: Privacy-Filter auf content + metadata-Felder.
            # User könnte sonst Credentials via title/description/source_url
            # in Metadata einschleusen (die ungefiltert nach Chroma gehen).
            _fields = {
                "content": content,
                "title": args.get("title") or "",
                "description": args.get("description") or "",
                "source_url": args.get("source_url") or "",
            }
            _filtered = {k: filter_content(v) for k, v in _fields.items()}

            # Jede rejection bricht den save (unclosed tag irgendwo im payload)
            for fname, fres in _filtered.items():
                if fres.rejected:
                    return [TextContent(type="text",
                        text=f"⚠️ [{fname}] {fres.rejection_reason}")]

            # Summaries sammeln für das Privacy-Badge in der Response
            _summary_bits = [
                f"{fname}: {fres.summary()}"
                for fname, fres in _filtered.items()
                if fres.had_secrets
            ]
            privacy_summary = (
                f" · 🔒 " + "; ".join(_summary_bits) if _summary_bits else ""
            )

            content = _filtered["content"].content
            title = _filtered["title"].content or None
            description = _filtered["description"].content or None
            source_url = _filtered["source_url"].content or None

            r = store.save(
                content, project,
                title=title,
                source_url=source_url,
                description=description,
                language=args.get("language", "auto"),
            )
            if r.get("error"):
                return [TextContent(type="text", text=f"⚠️ {r['error']}")]
            chunks_info = f", {r['chunks']} Chunk(s)" if r.get("chunks", 1) > 1 else ""
            graph_info = ""
            if r.get("graph") and isinstance(r["graph"], dict):
                g = r["graph"]
                if g.get("entities") or g.get("relations"):
                    graph_info = f", {g.get('entities', 0)} Entitäten, {g.get('relations', 0)} Relationen"
            return [TextContent(
                type="text",
                text=f"✅ Gespeichert (ID: {r['id']}, Projekt: {r['project']}{chunks_info}{graph_info}){privacy_summary}"
            )]
        else:
            r = store.forget(content, project)
            return [TextContent(type="text", text=f"{'✅ Gelöscht' if r['deleted'] else '⚠️ Nicht gefunden'} (ID: {r['id']})")]

    elif name == "recall":
        query = args.get("query", "")
        project = args.get("project", "default")
        incl_profile = args.get("include_profile", True)
        n = args.get("n_results", 15)

        if not query:
            return [TextContent(type="text", text="Fehler: query erforderlich")]

        mems = store.recall(query, project, n)
        parts = []

        if incl_profile:
            cached = store.get_profile(project)
            if cached:
                profile = cached
            else:
                all_mems = store.get_all(project)
                if all_mems:
                    profile = await get_engine().generate(all_mems)
                    store.set_profile(project, profile)
                else:
                    profile = "Keine Memories."
            parts.append("## User Profile\n" + profile + "\n")

        parts.append("## Relevante Memories\n")
        if mems:
            for i, m in enumerate(mems, 1):
                # Score-Zeile: absolute + normalised nebeneinander
                score_line = f"sim {m['similarity']}%"
                if "normalised_score" in m:
                    score_line += f" · rel {m['normalised_score']}/100"
                # Metadata-Zeile (nur wenn vorhanden)
                meta_bits = []
                if m.get("title"):
                    meta_bits.append(f"**{m['title']}**")
                if m.get("source_url"):
                    meta_bits.append(f"[Quelle]({m['source_url']})")
                if m.get("description"):
                    meta_bits.append(f"_{m['description']}_")
                header = f"### {i}. [{score_line}]"
                if meta_bits:
                    header += " · " + " · ".join(meta_bits)
                parts.append(header + "\n")
                parts.append(m["content"] + "\n")
                # Matched Chunk als Info-Footer, wenn Memory >1 Chunk hat
                if m.get("matched_chunk") and m.get("chunk_count", 1) > 1:
                    idx = m.get("matched_chunk_index", "?")
                    parts.append(
                        f"\n> *Bester Match: Chunk {idx}/{m['chunk_count']}*\n"
                    )
        else:
            parts.append("Keine gefunden.\n")

        return [TextContent(type="text", text="\n".join(parts))]

    elif name == "list_projects":
        projects = store.list_projects()
        lines = ["## Projekte\n"]
        for p in projects:
            lines.append(f"- **{p['name']}**: {p['count']} Memories")
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "stats":
        s = store.stats()
        text = f"""## Statistiken
**Gesamt:** {s['total_memories']} Memories ({s['total_chunks']} Chunks) in {s['projects']} Projekten
**Speicher:** `{s['path']}`
**Embeddings:** {s['embedding_provider']}
**Knowledge Graph:** {s['graph']['entities']} Entitäten, {s['graph']['relations']} Relationen, {s['graph']['memory_links']} Memory-Links
"""
        if s['graph']['entity_types']:
            text += f"**Entitäts-Typen:** {', '.join(s['graph']['entity_types'])}\n"
        if s['graph']['relation_types']:
            text += f"**Relations-Typen:** {', '.join(s['graph']['relation_types'])}\n"
        text += "\n"
        for proj, cnt in s['by_project'].items():
            text += f"- {proj}: {cnt}\n"
        return [TextContent(type="text", text=text)]

    elif name == "whoami":
        mems = store.get_all("default")[:10]
        if mems:
            profile = await get_engine().generate(mems)
            return [TextContent(type="text", text=f"## Benutzer-Info\n{profile}")]
        return [TextContent(type="text", text="Keine Memories vorhanden.")]

    elif name == "rechunk":
        project = args.get("project")
        r = store.rechunk_all(project=project)
        if r.get("error"):
            return [TextContent(type="text", text=f"⚠️ {r['error']}")]
        return [TextContent(
            type="text",
            text=(
                f"✅ {r['migrated']} Memories neu gechunkt "
                f"(Projekte: {r['projects']}, Provider: {r['provider']})"
            )
        )]

    elif name == "recall_multi":
        from .rag import recall_multi as _recall_multi
        query = args.get("query", "")
        project = args.get("project", "default")
        n = args.get("n_results", 15)
        expand_n = args.get("expand_n", 3)
        if not query:
            return [TextContent(type="text", text="Fehler: query erforderlich")]

        result = await _recall_multi(store, query, project, n, expand_n)
        mems = result["memories"]
        queries = result["queries"]

        parts = []
        if len(queries) > 1:
            parts.append("## Verwendete Queries")
            for i, q in enumerate(queries, 1):
                marker = " *(original)*" if i == 1 else ""
                parts.append(f"{i}. {q}{marker}")
            parts.append("")

        parts.append("## Relevante Memories")
        if mems:
            for i, m in enumerate(mems, 1):
                score_line = f"sim {m['similarity']}%"
                if "normalised_score" in m:
                    score_line += f" · rel {m['normalised_score']}/100"
                if m.get("matched_query") and m["matched_query"] != queries[0]:
                    mq = m["matched_query"][:60]
                    score_line += f' · via "{mq}"'
                header = f"### {i}. [{score_line}]"
                if m.get("title"):
                    header += f" · **{m['title']}**"
                parts.append(header)
                parts.append(m["content"])
                if m.get("matched_chunk") and m.get("chunk_count", 1) > 1:
                    idx = m.get("matched_chunk_index", "?")
                    parts.append(
                        f"\n> *Bester Match: Chunk {idx}/{m['chunk_count']}*"
                    )
                parts.append("")
        else:
            parts.append("Keine gefunden.")

        return [TextContent(type="text", text="\n".join(parts))]

    elif name == "answer":
        from .rag import recall_multi as _recall_multi, get_rag_engine
        question = args.get("question", "")
        project = args.get("project", "default")
        n = args.get("n_context", 8)
        use_mq = args.get("use_multi_query", True)
        if not question:
            return [TextContent(type="text", text="Fehler: question erforderlich")]

        # Retrieval
        if use_mq:
            r = await _recall_multi(store, question, project, n)
            memories = r["memories"]
            queries = r["queries"]
        else:
            memories = store.recall(question, project, n)
            queries = [question]

        # RAG-Antwort
        rag_result = await get_rag_engine().answer(question, memories)

        parts = [
            f"## Antwort\n{rag_result['answer']}",
            f"\n## Begründung\n{rag_result['justification']}",
        ]

        if len(queries) > 1:
            parts.append("\n## Query-Expansion")
            for q in queries:
                parts.append(f"- {q}")

        if memories:
            parts.append(f"\n## Quellen ({len(memories)} Memories)")
            for i, m in enumerate(memories, 1):
                label = m.get("title") or m["id"]
                parts.append(
                    f"{i}. [{m.get('similarity', 0)}% / "
                    f"rel {m.get('normalised_score', 0)}] {label}"
                )

        parts.append(f"\n_Provider: {rag_result.get('provider', '?')}_")
        return [TextContent(type="text", text="\n".join(parts))]

    # ── Progressive Disclosure (Phase 1) ─────────────────────
    elif name == "recall_index":
        query = args.get("query", "")
        project = args.get("project") or "default"
        n = args.get("n_results", 15)
        mem_type = args.get("mem_type")
        if not query:
            return [TextContent(type="text", text="Fehler: query erforderlich")]

        coll = store._get_collection(project)
        # embedder=None → Collection embedded via registrierter Ollama-EF
        # (oder Chroma-Default-EF, falls Ollama nicht aktiv).
        # project-Filter entfällt: eine Collection pro Projekt.
        hits = build_index(
            collection=coll,
            query=query,
            embedder=None,
            n_results=n,
            project=None,
            mem_type=mem_type,
        )
        payload = {
            "count": len(hits),
            "query": query,
            "project": project,
            "mem_type_filter": mem_type,
            "hits": [h.as_dict() for h in hits],
            "_next_step": (
                "Call recall_by_ids(ids=[...]) with the memory_ids you want "
                "full details for. Usually 2–5 IDs are enough."
            ),
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]

    elif name == "recall_by_ids":
        ids = args.get("ids") or []
        project = args.get("project") or "default"
        if not ids:
            return [TextContent(type="text", text="Fehler: ids erforderlich")]
        if len(ids) > 20:
            ids = ids[:20]

        coll = store._get_collection(project)

        # SQLite holds the unsplit Volltext per memory_id — use it to avoid
        # lossy chunk-reconstruction from Chroma. One prepared statement,
        # then the loader is called once per requested id.
        import sqlite3 as _sqlite3
        _conn = _sqlite3.connect(store.db_path)
        _rows = _conn.execute(
            f"SELECT id, content FROM memories WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        _conn.close()
        _content_map = {row[0]: row[1] for row in _rows}

        full = fetch_by_ids(
            coll, ids, project=None,
            content_loader=lambda mid: _content_map.get(mid),
        )
        payload = {
            "count": len(full),
            "requested": len(ids),
            "memories": [m.as_dict() for m in full],
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]

    # ── Knowledge Graph Tools ────────────────────────────────
    elif name == "graph_add_entity":
        r = store.graph.add_entity(
            name=args["name"],
            entity_type=args.get("type", "unknown"),
            properties=args.get("properties", {})
        )
        return [TextContent(type="text",
            text=f"✅ Entität: **{r['name']}** (Typ: {r['type']}, ID: {r['id']})")]

    elif name == "graph_add_relation":
        r = store.graph.add_relation(
            source_name=args["source"],
            target_name=args["target"],
            relation_type=args["relation_type"],
            source_type=args.get("source_type", "unknown"),
            target_type=args.get("target_type", "unknown"),
            properties=args.get("properties", {})
        )
        return [TextContent(type="text",
            text=f"✅ Relation: **{r['source']['name']}** —[{r['relation_type']}]→ **{r['target']['name']}**")]

    elif name == "graph_link_memory":
        r = store.graph.link_memory(
            memory_id=args["memory_id"],
            entity_name=args["entity_name"],
            entity_type=args.get("entity_type", "unknown"),
            project=args.get("project", "default")
        )
        return [TextContent(type="text",
            text=f"✅ Memory {r['memory_id']} ↔ Entität **{r['entity']['name']}**")]

    elif name == "graph_query":
        return [TextContent(type="text", text=_handle_graph_query(args))]

    elif name == "graph_stats":
        gs = store.graph.graph_stats()
        text = f"""## Knowledge Graph
**Entitäten:** {gs['entities']}
**Relationen:** {gs['relations']}
**Memory-Links:** {gs['memory_links']}
**Entitäts-Typen:** {', '.join(gs['entity_types']) if gs['entity_types'] else '—'}
**Relations-Typen:** {', '.join(gs['relation_types']) if gs['relation_types'] else '—'}"""
        return [TextContent(type="text", text=text)]

    return [TextContent(type="text", text=f"Unbekannt: {name}")]


# ═══════════════════════════════════════════════════════════════
# Graph Query Handler
# ═══════════════════════════════════════════════════════════════

def _handle_graph_query(args: dict) -> str:
    action = args["action"]
    g = store.graph

    if action == "find_connected":
        entity = args.get("entity", "")
        if not entity:
            return "Fehler: 'entity' erforderlich"
        r = g.find_connected(
            entity, max_depth=args.get("max_depth", 2),
            relation_type=args.get("relation_type")
        )
        if "error" in r:
            return f"⚠️ {r['error']}"
        lines = [f"## Verbunden mit '{r['root']}' ({r['total_nodes']} Knoten, {r['total_edges']} Kanten)\n"]
        for node in r["nodes"]:
            e = node["entity"]
            depth_marker = "  " * node["depth"] + ("→ " if node["depth"] > 0 else "● ")
            lines.append(f"{depth_marker}**{e['name']}** ({e.get('type', '?')}) [Tiefe {node['depth']}]")
        return "\n".join(lines)

    elif action == "shortest_path":
        fr = args.get("from_entity", "")
        to = args.get("to_entity", "")
        if not fr or not to:
            return "Fehler: 'from_entity' und 'to_entity' erforderlich"
        r = g.find_shortest_path(fr, to, max_depth=args.get("max_depth", 5))
        if r.get("length", -1) < 0:
            return f"⚠️ Kein Pfad zwischen '{fr}' und '{to}'"
        lines = [f"## Pfad: {fr} → {to} (Länge: {r['length']})\n"]
        for i, step in enumerate(r["path"]):
            e = step["entity"]
            rel = step.get("via_relation")
            if rel:
                lines.append(f"  —[{rel}]→ **{e['name']}** ({e.get('type', '?')})")
            else:
                lines.append(f"● **{e['name']}** ({e.get('type', '?')})")
        return "\n".join(lines)

    elif action == "subgraph":
        entities = args.get("entities", [])
        if not entities:
            return "Fehler: 'entities' Liste erforderlich"
        r = g.get_subgraph(entities, max_depth=args.get("max_depth", 1))
        lines = [f"## Subgraph ({r['total_nodes']} Knoten, {r['total_edges']} Kanten)\n"]
        lines.append("### Knoten")
        for node in r["nodes"]:
            e = node["entity"]
            lines.append(f"- **{e['name']}** ({e.get('type', '?')})")
        lines.append("\n### Kanten")
        for edge in r["edges"]:
            lines.append(f"- {edge['from'][:8]}… —[{edge['type']}]→ {edge['to'][:8]}…")
        return "\n".join(lines)

    elif action == "relations":
        entity = args.get("entity", "")
        if not entity:
            return "Fehler: 'entity' erforderlich"
        rels = g.get_relations(
            entity_name=entity,
            relation_type=args.get("relation_type")
        )
        if not rels:
            return f"Keine Relationen für '{entity}'"
        lines = [f"## Relationen von '{entity}' ({len(rels)})\n"]
        for rel in rels:
            if rel["direction"] == "out":
                lines.append(f"  → [{rel['relation_type']}] → **{rel.get('target_name', '?')}**")
            else:
                lines.append(f"  ← [{rel['relation_type']}] ← **{rel.get('source_name', '?')}**")
        return "\n".join(lines)

    elif action == "search":
        query = args.get("query", "")
        if not query:
            return "Fehler: 'query' erforderlich"
        results = g.search_entities(query, entity_type=args.get("entity_type"))
        if not results:
            return f"Keine Entitäten für '{query}'"
        lines = [f"## Entitäten-Suche: '{query}' ({len(results)} Treffer)\n"]
        for e in results:
            lines.append(f"- **{e['name']}** (Typ: {e['type']}, ID: {e['id']})")
        return "\n".join(lines)

    elif action == "entity_memories":
        entity = args.get("entity", "")
        if not entity:
            return "Fehler: 'entity' erforderlich"
        mems = g.get_entity_memories(entity_name=entity)
        if not mems:
            return f"Keine Memories für Entität '{entity}'"
        lines = [f"## Memories verknüpft mit '{entity}' ({len(mems)})\n"]
        for m in mems:
            lines.append(f"- Memory `{m['memory_id']}` (Projekt: {m['project']})")
        return "\n".join(lines)

    return f"Unbekannte Graph-Action: {action}"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    asyncio.run(run())

async def run():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

if __name__ == "__main__":
    main()
