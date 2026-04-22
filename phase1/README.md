# supermemory.granaria — Phase 1 Upgrade

Drop-in extensions for your existing `supermemory.granaria` FastMCP server,
inspired by (but not copied from) claude-mem's progressive-disclosure and
dual-tag privacy patterns.

**Status:** 31/31 tests green, ready for integration.

---

## What this adds

| Module | Purpose | Risk |
|---|---|---|
| `tools/recall_progressive.py` | 3-layer search workflow (`recall_index` + `recall_by_ids`) — ~5–10× token saving on broad queries | LOW — purely additive, no changes to existing tools |
| `hooks/privacy_filter.py` | Edge-layer content sanitation — `<private>` / `<secret>` tag stripping + credential pattern redaction | MEDIUM — wraps the existing `memory` endpoint. Reversible by removing the decorator. |

Both modules are **standalone**. You can deploy either one independently.

---

## File layout (to copy into your server)

```
/Applications/ServBay/www/supermemory.granaria/
├── phase1/
│   ├── __init__.py                  (empty, makes it a package)
│   ├── tools/
│   │   ├── __init__.py
│   │   └── recall_progressive.py   ← new
│   ├── hooks/
│   │   ├── __init__.py
│   │   └── privacy_filter.py       ← new
│   └── tests/
│       └── test_phase1.py
└── app.py  (or whatever your main FastMCP file is called — patch below)
```

---

## Install

```bash
cd /Applications/ServBay/www/supermemory.granaria

# 1. Copy in the module
cp -r /path/to/phase1 ./phase1
touch phase1/__init__.py phase1/tools/__init__.py phase1/hooks/__init__.py

# 2. Run the tests against YOUR Python (should use the venv that the
#    LaunchAgent ac.granaria.supermemory uses — Python 3.13)
source .venv/bin/activate  # or wherever your venv is
python3 phase1/tests/test_phase1.py
# Expected: Ran 31 tests in ~0.003s  OK

# 3. Patch your main server file — see "Integration patch" below.

# 4. Validate syntax without starting the server:
python3 -m py_compile app.py

# 5. Restart the LaunchAgent
launchctl kickstart -k gui/$(id -u)/ac.granaria.supermemory

# 6. Verify in logs (replace with your actual log path):
tail -f ~/Library/Logs/ac.granaria.supermemory.log
# Look for: "Registered progressive recall tools: recall_index, recall_by_ids"
```

---

## Integration patch

In your main server file (e.g. `app.py`), add these lines near where you
currently register your existing `memory` / `recall` tools:

```python
# --- Phase 1 imports ---
from phase1.tools.recall_progressive import register_progressive_tools
from phase1.hooks.privacy_filter import filter_content

# --- Register progressive recall (Baustein 1) ---
register_progressive_tools(mcp, chroma_client, embedder)

# --- Wire privacy filter into existing memory handler (Baustein 2) ---
# Option A: if `memory` is a plain function, use the decorator
@mcp.tool()
def memory(content: str, project: str = "default", **kwargs):
    filtered = filter_content(content, strict_unclosed=True)
    if filtered.rejected:
        return {"ok": False, "error": filtered.rejection_reason}

    # >>> your existing memory-storage logic goes here <<<
    # Use `filtered.content` instead of the raw `content`:
    memory_id = store_memory(
        content=filtered.content,
        project=project,
        **kwargs,
    )

    return {
        "ok": True,
        "memory_id": memory_id,
        "privacy": {"summary": filtered.summary(),
                    "secrets_found": filtered.had_secrets},
    }
```

**Don't use the `wrap_memory_handler` decorator** if your `memory` function
already does non-trivial argument handling (project-based collection routing,
embedding, knowledge-graph linking, …). The inline `filter_content(...)` call
is clearer and equally safe.

---

## Rollback

Both changes are fully reversible.

```bash
# Progressive recall — remove the register line, restart:
sed -i.bak '/register_progressive_tools/d' app.py
launchctl kickstart -k gui/$(id -u)/ac.granaria.supermemory

# Privacy filter — remove the filter_content call from memory handler,
# restart. Data already persisted stays as-is (no schema changes).
```

No DB migrations run. No existing memory is touched.

---

## What's NOT in Phase 1

For reference, these are on the roadmap for Phase 2 (not yet built):

- **Content-hash deduplication** — SHA-256 of normalised content, skip insert on match
- **Timeline sequence table** — previous/next memory relations for causal queries
- **Live web-viewer UI inspiration** — visual design polish

These become interesting only once Phase 1 has been live for 2+ weeks and
you have a feel for whether progressive recall actually gets used.

---

## Licence note

All code in this directory is written from scratch. No code was copied from
`thedotmack/claude-mem` (AGPL-3.0). Only the high-level design patterns
(progressive disclosure, dual-tag privacy) are inspired by that project's
public documentation. Design patterns are not copyrightable; the
implementation is your own.

You may licence this module under whatever terms suit your supermemory.granaria
project — MIT, proprietary, AGPL, your choice.
