# MCP Client Compatibility

`supermemory.granaria` is a standards-conformant MCP server (stdio, reference
`mcp.server.Server` API). Nothing in the protocol is Claude-specific, so any
compliant MCP client should be able to mount it.

## Compatibility matrix

| Client | Status | Notes |
|---|---|---|
| Claude Desktop / Claude Code | ✅ primary target | `claude_desktop_config.json` as in the README |
| GitHub Copilot (VS Code) | ✅ expected | native MCP since 2025; `.vscode/mcp.json` or user settings; same `command`/`args` shape as Claude |
| Copilot CLI | ✅ expected | `copilot mcp add supermemory.granaria -- python3 -m local_supermemory.server` |
| Cursor / Windsurf / Cline / Continue.dev | ✅ expected | per-client UI, same config shape |
| Gemini CLI | ✅ expected | `activate_skill` + MCP auto-discovery |
| Grok (xAI) | ⚠️ unverified | No official MCP client in Grok Desktop at the time of writing. Grok via API would need a custom MCP-to-function-calling bridge. |

Only Claude is end-to-end verified in production here. Entries marked
"expected" are based on each client's documented MCP support, not first-hand
testing in this repo.

## Non-obvious behaviours for non-Claude clients

### Privacy-tag convention

The privacy filter runs server-side on every `memory` save, so tier-2 pattern
redaction (AWS/Stripe/GitHub/...) is automatic for any client.

Tier-1 `<private>…</private>` stripping is only useful if the client-side LLM
knows to wrap sensitive content in those tags. Claude learns this through
`phase1/SKILL_PATCH.md`. For other clients, add a user instruction such as:

> Wrap anything that should never be persisted in `<private>…</private>`
> tags. Unclosed tags will cause the save to be rejected.

Put this in the client's equivalent of system prompt / custom instructions
(`.github/copilot-instructions.md` for Copilot, profile prompt for Gemini,
etc.).

### Progressive recall

No special instruction is strictly needed — the tool descriptions explicitly
tell the LLM to call `recall_index` first and fetch with `recall_by_ids` only
for IDs it actually needs. Any capable model should pick this up from
descriptions alone. If you see a client consistently defaulting to the plain
`recall` tool and wasting tokens, add:

> For memory search, prefer `recall_index` first; only use `recall_by_ids`
> for 2–5 specific IDs you identified in the index.

### Ollama features

`answer`, `recall_multi`, and high-quality embeddings require a local Ollama
install. Without Ollama the server degrades gracefully to ChromaDB default
embeddings and keyword-based profile fallbacks. This is independent of the
MCP client.

## Known gaps

- No load testing against clients other than Claude.
- `phase1/SKILL_PATCH.md` is written in Claude voice; adapt its phrasing
  when porting to another client's instruction channel.
- Tool descriptions are in English/German mix — some clients may prefer
  consistent-language descriptions for tool selection quality.

## When in doubt

Start the server manually and inspect its tool list:

```bash
python3 -m local_supermemory.server <<< '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

If that returns 15 tools cleanly, the MCP surface is healthy and any client
failure is a client-side config issue rather than a server bug.
