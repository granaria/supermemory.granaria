# Phase 2 — Open Items

Tracking file for Phase 2 bausteine that are discussed but not yet built.
Committed here so the plan doesn't live only in chat history.

## Baustein D — Tool-Call Audit + Health Checks

**Status:** noted, not started.
**Motivation:** silent parameter loss (e.g. `project="automation"` gets dropped
and the save lands in `default`) is the worst failure mode because nothing
errors out — the user only notices much later when `list_projects` looks
wrong. Observed informally during the Phase 1 / dashboard session on
2026-04-22; reproduction path not isolated (could be client-side tool-call
formatting, could be server-side arg handling). The right fix is not to
hunt the specific bug but to make the whole dispatch path observable.

### Server side (transparent — no public-API change)

1. Structured tool-call log appended to
   `~/.granaria.supermemory/tool_audit.jsonl`:
   - `tool_name`, `ts`, `duration_ms`
   - `args_received` (exactly what the dispatcher got)
   - `args_types` (type per key, for coercion-issue visibility)
   - `result_summary` (first sentence / first 200 chars of response)
2. Schema validation warning: each call validated against its `inputSchema`;
   missing required fields or type mismatches → warning entry, but the
   handler still runs (don't paper over bugs — expose them).
3. Silent-default detection for `memory`: when `project` is absent in
   `args_received` the audit entry marks `project_defaulted=true`. Cheap
   and specifically catches the scenario that motivated this baustein.

### Surface

- New MCP tool `diagnostics_recent(n=20)` → last N tool-calls with full
  args + result summaries.
- Dashboard panel "Tool Calls" next to the existing "Filter Events" panel,
  filterable by tool name, shows received-vs-schema diff.
- Plain file viewer: users can also tail the JSONL directly.

### Non-goals

- No auto-correction. If an arg is missing, log it, let the handler use
  its default — but make the event visible. Auto-correct hides bugs.
- No full request-body logging for large contents. Length + SHA-8
  fingerprint only (aligned with how privacy_filter already audits).
- No performance instrumentation beyond `duration_ms` — this is a
  debug/observability tool, not an APM.

### Shape (pre-implementation estimate)

- New module `phase1/dashboard/tool_audit.py` (~150 LOC)
- One-line hook in `local_supermemory/server.py::call_tool` (decorator
  or prepend in dispatcher)
- New MCP tool definition + dispatcher branch (~30 LOC)
- Dashboard panel: new `/api/tool_calls` endpoint (~30 LOC),
  frontend panel in `index.html` (~80 LOC JS/HTML)
- 8–10 unit tests (dispatch logging, schema mismatch, silent-default
  detection, fingerprint behavior)
- Section in `phase1/README.md`
- Entry in dashboard Tool-Calls panel help text

Total: ~400 LOC, one focused commit.

### Decision point

Keep on backlog; build when either (a) we see the parameter-loss
symptom reproducibly, or (b) before onboarding another MCP client
where we want the transparency from day 1.

---

## Other Bausteine

Other Phase 2 items exist in prior planning but were not discussed in
detail in the Phase 1 / dashboard session. Add sections here as they
get concretized.
