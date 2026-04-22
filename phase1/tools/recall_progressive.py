"""
Progressive Disclosure Recall for supermemory.granaria
=======================================================

Inspired by claude-mem's 3-layer workflow (thedotmack/claude-mem, AGPL-3.0),
reimplemented from scratch for the Granaria FastMCP server.

Concept (NOT code) reused:
    Layer 1: recall_index(query)           → tiny rows (id, title, score)
    Layer 2: recall_by_ids(ids)            → full payload for selected IDs

Token math:
    Old `recall(n=15)` with full payloads  ≈ 8–15 k tokens
    New index-first pattern                ≈ 400–800 tokens for index,
                                             + on-demand fetch for 2–3 hits
                                           ≈ 1.5–3 k tokens total
    → ~5–10× saving on typical queries.

Integration
-----------
Assumes your existing FastMCP server exposes a Chroma collection per project
and that each memory document has metadata fields:
    {
        "memory_id":  str (unique),
        "title":      str (first ~80 chars or explicit title),
        "created_at": ISO-8601 str,
        "mem_type":   str (e.g. "note"|"decision"|"fix"|"fact"|"code"),
        "project":    str,
    }

If any of those fields are missing, `_safe_meta()` degrades gracefully.

Register in your main server:
    from tools.recall_progressive import register_progressive_tools
    register_progressive_tools(mcp, chroma_client, embedder)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable

log = logging.getLogger("supermemory.progressive")


# --------------------------------------------------------------------------
#  Data classes
# --------------------------------------------------------------------------

@dataclass(slots=True)
class IndexHit:
    """Layer-1 result row. Kept deliberately tiny."""
    memory_id: str
    title: str
    mem_type: str
    created_at: str
    score: float
    project: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FullMemory:
    """Layer-2 result row. Full payload for gezielte IDs."""
    memory_id: str
    content: str
    title: str
    mem_type: str
    created_at: str
    project: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
#  Core logic
# --------------------------------------------------------------------------

def _safe_meta(meta: dict[str, Any] | None, key: str, default: str = "") -> str:
    """Robust metadata access — legacy memories may miss fields."""
    if not meta:
        return default
    v = meta.get(key, default)
    return v if isinstance(v, str) else (str(v) if v is not None else default)


def _derive_title(content: str, max_len: int = 80) -> str:
    """Fallback title: first line, or first max_len chars."""
    first_line = (content or "").strip().split("\n", 1)[0]
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 1] + "…"


def build_index(
    collection,
    query: str,
    embedder,
    n_results: int = 15,
    project: str | None = None,
    mem_type: str | None = None,
) -> list[IndexHit]:
    """
    Layer 1 — Return a slim index of matching memories.

    This is the function Claude should call FIRST. It returns just enough
    information to decide which memories deserve a full fetch.

    Parameters
    ----------
    collection : chromadb.Collection
    query      : natural-language query
    embedder   : any object with .encode(str) → list[float]
    n_results  : max hits (default 15 — same as existing `recall`)
    project    : optional project filter (uses Chroma where-clause)
    mem_type   : optional type filter ("decision", "fix", etc.)

    Returns
    -------
    List[IndexHit] sorted by score descending.
    """
    if not query or not query.strip():
        return []

    # Build Chroma where-filter (None if no filter → don't pass it)
    where: dict[str, Any] | None = None
    filters = []
    if project:
        filters.append({"project": project})
    if mem_type:
        filters.append({"mem_type": mem_type})
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    t0 = time.perf_counter()
    try:
        if embedder is None:
            result = collection.query(
                query_texts=[query.strip()],
                n_results=max(1, min(n_results, 50)),
                where=where,
                include=["metadatas", "distances", "documents"],
            )
        else:
            query_embedding = embedder.encode(query.strip())
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, min(n_results, 50)),
                where=where,
                include=["metadatas", "distances", "documents"],
            )
    except Exception as e:
        log.exception("Chroma query failed for %r: %s", query, e)
        return []
    dt = (time.perf_counter() - t0) * 1000

    ids = result.get("ids", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    docs = result.get("documents", [[]])[0]

    hits: list[IndexHit] = []
    for chroma_id, meta, dist, doc in zip(ids, metas, dists, docs):
        # Prefer explicit memory_id in metadata, fall back to chroma id
        memory_id = _safe_meta(meta, "memory_id", default=chroma_id or "")
        if not memory_id:
            continue

        title = _safe_meta(meta, "title") or _derive_title(doc or "")
        # Chroma returns squared L2 distance (or cosine-dist, depending on
        # collection setup). We normalise to a 0..1 similarity where higher = better.
        score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0

        hits.append(
            IndexHit(
                memory_id=memory_id,
                title=title,
                mem_type=_safe_meta(meta, "mem_type", default="note"),
                created_at=_safe_meta(meta, "created_at"),
                score=round(score, 4),
                project=_safe_meta(meta, "project", default="default"),
            )
        )

    log.info(
        "recall_index q=%r n=%d project=%s → %d hits (%.1f ms)",
        query[:40], n_results, project, len(hits), dt,
    )
    return hits


def fetch_by_ids(
    collection,
    memory_ids: list[str],
    project: str | None = None,
    content_loader: Callable[[str], str | None] | None = None,
) -> list[FullMemory]:
    """
    Layer 2 — Fetch full payloads for a list of memory IDs.

    Claude should only call this with IDs it deemed relevant from Layer 1.

    Parameters
    ----------
    content_loader : optional callable mid -> full text.
        If given, the content of each returned FullMemory is loaded via this
        callback (typically a SQLite lookup in the host store). This is the
        robust path for stores that split memories into overlapping chunks:
        reconstructing a clean Volltext from Chroma chunks is lossy, while
        a SQLite row holds the unsplit original. If `None` (default),
        chunks are deduplicated by `memory_id` and the first chunk's content
        is used as a best-effort fallback.
    """
    if not memory_ids:
        return []

    # De-dupe while preserving order
    seen: set[str] = set()
    unique_ids = [mid for mid in memory_ids if not (mid in seen or seen.add(mid))]

    # ── Strategy 1: direct Chroma document-id match ──────────────────
    # Works for the common case where memory_id == chroma document id
    # (e.g. legacy entries without a `memory_id` metadata field, or
    # stores that deliberately use the same value for both).
    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    ids: list[str] = []
    try:
        r1 = collection.get(ids=unique_ids, include=["documents", "metadatas"])
        ids = list(r1.get("ids") or [])
        docs = list(r1.get("documents") or [])
        metas = list(r1.get("metadatas") or [])
    except Exception as e:
        log.debug("fetch_by_ids: ids-lookup failed (%s), trying metadata where", e)

    # ── Strategy 2: metadata `memory_id` where-clause fallback ───────
    # Only run for IDs still missing after Strategy 1.
    missing = [
        mid for mid in unique_ids
        if mid not in ids
        and not any(_safe_meta(m, "memory_id") == mid for m in metas)
    ]
    if missing:
        where: dict[str, Any] = {"memory_id": {"$in": missing}}
        if project:
            where = {"$and": [where, {"project": project}]}
        try:
            r2 = collection.get(where=where, include=["documents", "metadatas"])
            ids.extend(r2.get("ids") or [])
            docs.extend(r2.get("documents") or [])
            metas.extend(r2.get("metadatas") or [])
        except Exception as e:
            log.exception("fetch_by_ids: metadata where-lookup failed: %s", e)

    # ── Group chunks by memory_id (one FullMemory per memory, not per chunk) ──
    # Multi-chunk memories produce multiple Chroma rows that all share the
    # same `memory_id` in metadata. Without this dedup the caller would get
    # N rows for a single requested ID.
    groups: dict[str, list[tuple[int, str, str, dict]]] = {}
    for chroma_id, doc, meta in zip(ids, docs, metas):
        mid = _safe_meta(meta, "memory_id", default=chroma_id or "")
        raw_idx = (meta or {}).get("chunk_index")
        try:
            idx = int(raw_idx) if raw_idx is not None else 0
        except (TypeError, ValueError):
            idx = 0
        groups.setdefault(mid, []).append((idx, chroma_id or "", doc or "", meta or {}))

    out: list[FullMemory] = []
    for mid, entries in groups.items():
        entries.sort(key=lambda x: x[0])  # by chunk_index
        first_meta = entries[0][3]
        # Content: authoritative loader if available; otherwise first chunk
        # (best-effort — chunk overlap means full reconstruction is lossy).
        content: str | None = None
        if content_loader is not None:
            try:
                content = content_loader(mid)
            except Exception as e:
                log.warning("content_loader failed for %s: %s", mid, e)
                content = None
        if content is None:
            content = entries[0][2]  # fallback: first chunk's doc
            if len(entries) > 1 and content_loader is None:
                log.info(
                    "fetch_by_ids: memory %s has %d chunks; returning first chunk "
                    "only (pass content_loader= for full Volltext)",
                    mid, len(entries),
                )
        out.append(
            FullMemory(
                memory_id=mid,
                content=content,
                title=_safe_meta(first_meta, "title") or _derive_title(content),
                mem_type=_safe_meta(first_meta, "mem_type", default="note"),
                created_at=_safe_meta(first_meta, "created_at"),
                project=_safe_meta(first_meta, "project", default="default"),
                metadata=first_meta,
            )
        )

    # Preserve requested order (Chroma may return in its own order)
    order = {mid: i for i, mid in enumerate(unique_ids)}
    out.sort(key=lambda m: order.get(m.memory_id, 1_000_000))

    log.info(
        "recall_by_ids: requested=%d returned=%d (from %d chunks)",
        len(unique_ids), len(out), len(ids),
    )
    return out


# --------------------------------------------------------------------------
#  FastMCP registration helper
# --------------------------------------------------------------------------

def register_progressive_tools(mcp, chroma_client, embedder, *, default_collection="memories"):
    """
    Register two new MCP tools on the given FastMCP instance.

    Usage in your main server file (e.g. app.py):

        from fastmcp import FastMCP
        from tools.recall_progressive import register_progressive_tools

        mcp = FastMCP("supermemory.granaria")
        # ... existing tool registrations ...
        register_progressive_tools(mcp, chroma_client, embedder)

    Both tools respect your existing project & embedding setup.
    """

    def _get_collection(project: str | None):
        """Resolve collection. Adapt this if you use one-collection-per-project."""
        # If you already have a helper like `get_or_create_collection(project)`,
        # swap this for it. Here we assume a single collection with `project`
        # stored in metadata.
        return chroma_client.get_or_create_collection(default_collection)

    @mcp.tool()
    def recall_index(
        query: str,
        n_results: int = 15,
        project: str | None = None,
        mem_type: str | None = None,
    ) -> dict[str, Any]:
        """Layer 1 of progressive recall.

        Returns a SLIM index of matching memories: id, title, type, timestamp,
        similarity score. Use this FIRST to scan what's available, then call
        `recall_by_ids(ids=[...])` to fetch full content only for the
        memories you actually need.

        This saves ~5–10× tokens compared to the full `recall` tool.

        Args:
            query: Natural-language search query.
            n_results: Max hits (default 15, max 50).
            project: Optional project filter, e.g. "granaria-vc".
            mem_type: Optional type filter, e.g. "decision", "fix", "fact".

        Returns:
            {"count": int, "hits": [IndexHit, ...]}
        """
        coll = _get_collection(project)
        hits = build_index(
            collection=coll,
            query=query,
            embedder=embedder,
            n_results=n_results,
            project=project,
            mem_type=mem_type,
        )
        return {
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

    @mcp.tool()
    def recall_by_ids(
        ids: list[str],
        project: str | None = None,
    ) -> dict[str, Any]:
        """Layer 2 of progressive recall.

        Fetch FULL content for specific memory IDs (as returned by
        `recall_index`). Use this only after narrowing down with the index.

        Args:
            ids: List of memory_id strings. Max 20 per call.
            project: Optional project filter for safety.

        Returns:
            {"count": int, "memories": [FullMemory, ...]}
        """
        if len(ids) > 20:
            ids = ids[:20]
            log.warning("recall_by_ids truncated to 20 IDs")
        coll = _get_collection(project)
        full = fetch_by_ids(coll, ids, project=project)
        return {
            "count": len(full),
            "requested": len(ids),
            "memories": [m.as_dict() for m in full],
        }

    log.info("Registered progressive recall tools: recall_index, recall_by_ids")
    return recall_index, recall_by_ids
