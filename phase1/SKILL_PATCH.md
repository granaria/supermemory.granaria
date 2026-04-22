# Supermemory Phase 1 — Agent Behaviour Changes

> Drop this into `/Applications/ServBay/www/supermemory.granaria/SKILL.md`
> (or append to the existing one). This is what tells Claude how to USE the
> new tools correctly.

---

## 1. Progressive Recall Workflow (Token Efficiency)

When a question may require looking up prior memories, prefer the **index-first pattern**:

1. **`recall_index(query, ...)`** — returns `{id, title, mem_type, created_at, score}` for up to 15 candidates. Typically 400–800 tokens total.
2. **Skim the titles & types.** Pick the 2–5 IDs that actually look relevant.
3. **`recall_by_ids(ids=[...])`** — fetches full content ONLY for those IDs.

### When to use `recall_index` + `recall_by_ids` (new pattern)
- Broad questions: *"Was wissen wir über Paarstraße 7?"*
- Exploratory queries where you're not yet sure what's relevant
- Any query where you'd historically ask for `n_results >= 10`

### When the single-call `recall` is still fine
- Specific factual lookups you expect 1–3 memories to answer: *"IP von VPS2?"*
- Queries where `n_results <= 3` is enough

### When `answer` / `recall_multi` is still preferred
- Questions that need LLM synthesis across multiple memories
- Ambiguous or paraphrased queries where expansion helps

### Heuristic
```
if "übersicht" / "zusammenfassung" / "alles zu ..." / broad topic:
    → recall_index first, then recall_by_ids for top 3
elif question is narrow factoid:
    → recall (single call, n_results=3)
elif question requires reasoning across memories:
    → answer
```

---

## 2. Privacy Tags (NEW)

When you (or the user) store content that may contain secrets, wrap the
sensitive portion:

```
<private>API_KEY=sk_live_abc123</private>
```

or for longer blocks:

```
<secret>
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----
</secret>
```

The filter runs at the **memory endpoint edge** — secrets are stripped before
they ever reach Chroma or SQLite. The returned response includes
`privacy.summary` with an audit trail (fingerprint, not the secret itself).

### Automatic pattern redaction (belt + suspenders)

Even WITHOUT tags, these patterns get redacted:
- AWS access keys (`AKIA…`, `ASIA…`)
- Stripe keys (`sk_live_…`, `sk_test_…`)
- GitHub tokens (`ghp_…`, `gho_…`)
- OpenAI legacy keys (`sk-…T3BlbkFJ…`)
- Anthropic keys (`sk-ant-…`)
- JSON Web Tokens (`eyJ…`)
- PEM private-key blocks

### What does NOT get auto-redacted (by design)
- Plain passwords like `password: hunter2` — too many false positives
- Short tokens, IP addresses, usernames — not universally secret
- **→ For those, USE `<private>` tags explicitly.**

### Rejection behaviour
An **unclosed** `<private>` or `<secret>` tag causes the memory to be
REJECTED with a clear error message. This prevents accidents where you
meant to wrap a secret but made a typo.

---

## 3. Migration guidance (for existing memories)

You don't need to do anything. Existing memories stay as-is. The filter
only applies to NEW writes via the `memory` endpoint.

If you want to audit old memories for accidentally-stored secrets:
```bash
cd /Applications/ServBay/www/supermemory.granaria
python3 -m phase1.migrations.audit_existing --project all --dry-run
```
(Audit script is optional — ship without it unless you actually need it.)

---

## 4. Cheat sheet for agents

| Task | Old call | New call |
|---|---|---|
| Broad topic scan | `recall(query, n=15)` | `recall_index(query) → recall_by_ids([...])` |
| Narrow factoid | `recall(query, n=3)` | unchanged |
| Multi-hop synthesis | `answer(question)` | unchanged |
| Store content with secret | `memory(content)` | wrap secret in `<private>…</private>` |
| Store ordinary content | `memory(content)` | unchanged |

---

## 5. Metrics to watch (first 2 weeks)

After deploy, watch the logs for:
- `supermemory.progressive` logger — count of `recall_index` vs old `recall` calls
- `supermemory.privacy` logger — WARNINGs indicate successful strips (good) or unclosed-tag rejections (investigate)
- Chroma query latency — should be unchanged or slightly lower (smaller result payloads)

If `recall_index` usage stays below ~30% of total recalls after 2 weeks, the
SKILL.md isn't convincing enough — iterate on wording.
