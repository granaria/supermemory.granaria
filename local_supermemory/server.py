"""Local Supermemory MCP Server"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .store import MemoryStore
from .profile import get_engine

server = Server("local-supermemory")
store = MemoryStore()

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory",
            description="Speichere oder vergesse Informationen. action: 'save'|'forget', content: Text, project: optional",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["save", "forget"], "default": "save"},
                    "content": {"type": "string", "maxLength": 200000},
                    "project": {"type": "string", "maxLength": 128}
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
        )
    ]

@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    if name == "memory":
        action = args.get("action", "save")
        content = args.get("content", "")
        project = args.get("project", "default")
        if not content:
            return [TextContent(type="text", text="Fehler: content erforderlich")]
        if action == "save":
            r = store.save(content, project)
            return [TextContent(type="text", text=f"✅ Gespeichert (ID: {r['id']}, Projekt: {r['project']})")]
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
                parts.append(f"### {i}. ({m['similarity']}%)\n{m['content']}\n")
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
**Gesamt:** {s['total']} Memories in {s['projects']} Projekten
**Speicher:** `{s['path']}`
"""
        for proj, cnt in s['by_project'].items():
            text += f"\n- {proj}: {cnt}"
        return [TextContent(type="text", text=text)]
    
    elif name == "whoami":
        mems = store.get_all("default")[:10]
        if mems:
            profile = await get_engine().generate(mems)
            return [TextContent(type="text", text=f"## Benutzer-Info\n{profile}")]
        return [TextContent(type="text", text="Keine Memories vorhanden.")]
    
    return [TextContent(type="text", text=f"Unbekannt: {name}")]

def main():
    asyncio.run(run())

async def run():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

if __name__ == "__main__":
    main()
