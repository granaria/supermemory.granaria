"""Server-side hook: apply dashboard config to a save request.

`local_supermemory/server.py` calls exactly ONE function from here:
`apply_save_policy(...)`. Everything else lives inside phase1/dashboard.

Responsibilities of the policy layer:
  1. Reject save if project is in the blocklist.
  2. Reject save if content exceeds configured max length.
  3. Run privacy-filter on content + title + description + source_url
     with the currently configured toggles.
  4. Propagate `auto_extract_graph` default to the caller.
  5. Append an audit event for UI display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phase1.hooks.privacy_filter import filter_content, FilterResult

from .config import get_config
from . import audit


@dataclass
class SavePolicyResult:
    rejected: bool
    rejection_reason: str
    # Post-filter values (None = caller should drop the field)
    content: str
    title: str | None
    description: str | None
    source_url: str | None
    # Cosmetics / audit
    had_secrets: bool
    per_field_summary: dict[str, str]
    badge: str                  # "" or " · 🔒 …" — append to success response
    # Capture-behavior passthrough
    auto_extract_graph: bool


def _ok_passthrough(content: str, title, description, source_url,
                    auto_extract: bool) -> SavePolicyResult:
    return SavePolicyResult(
        rejected=False, rejection_reason="",
        content=content, title=title, description=description,
        source_url=source_url,
        had_secrets=False, per_field_summary={}, badge="",
        auto_extract_graph=auto_extract,
    )


def _reject(reason: str, content: str, title, description, source_url) -> SavePolicyResult:
    return SavePolicyResult(
        rejected=True, rejection_reason=reason,
        content=content, title=title, description=description,
        source_url=source_url,
        had_secrets=False, per_field_summary={}, badge="",
        auto_extract_graph=False,
    )


def apply_save_policy(
    *,
    content: str,
    project: str,
    title: str | None = None,
    description: str | None = None,
    source_url: str | None = None,
) -> SavePolicyResult:
    cfg = get_config().get()
    priv = cfg["privacy"]
    cap = cfg["capture"]
    auto_extract = bool(cap.get("auto_extract_graph", True))

    # 1) Project blocklist
    if project in (cap.get("blocked_projects") or []):
        audit.log({
            "event": "save_blocked_project",
            "project": project,
            "content_len": len(content),
        })
        return _reject(
            f"Project '{project}' is blocked via dashboard config.",
            content, title, description, source_url,
        )

    # 2) Size limit
    max_chars = int(cap.get("max_content_chars") or 0)
    if max_chars > 0 and len(content) > max_chars:
        audit.log({
            "event": "save_rejected_too_large",
            "project": project,
            "size": len(content),
            "limit": max_chars,
        })
        return _reject(
            f"Content length {len(content)} exceeds configured limit {max_chars}.",
            content, title, description, source_url,
        )

    # 3) Privacy filter per field
    enabled = bool(priv.get("enabled", True))
    strip_tags = bool(priv.get("strip_private_tags", True))
    strict = bool(priv.get("strict_unclosed_tags", True))
    pattern_toggles = priv.get("patterns") or {}

    fields = {
        "content": content,
        "title": title or "",
        "description": description or "",
        "source_url": source_url or "",
    }
    results: dict[str, FilterResult] = {}
    for k, v in fields.items():
        results[k] = filter_content(
            v,
            strict_unclosed=strict,
            enabled=enabled,
            strip_private_tags=strip_tags,
            enabled_patterns=pattern_toggles,
        )

    # Any unclosed-tag rejection?
    for fname, fres in results.items():
        if fres.rejected:
            audit.log({
                "event": "save_rejected_unclosed_tag",
                "project": project,
                "field": fname,
            })
            return _reject(
                f"[{fname}] {fres.rejection_reason}",
                content, title, description, source_url,
            )

    # Aggregate summaries for the response badge
    summaries = {k: r.summary() for k, r in results.items() if r.had_secrets}
    had_secrets = bool(summaries)
    badge = ""
    if summaries:
        badge = " · 🔒 " + "; ".join(f"{k}: {v}" for k, v in summaries.items())

    audit.log({
        "event": "save_ok",
        "project": project,
        "had_secrets": had_secrets,
        "summaries": summaries,
        "content_len": len(results["content"].content),
    })

    return SavePolicyResult(
        rejected=False, rejection_reason="",
        content=results["content"].content,
        title=(results["title"].content or None),
        description=(results["description"].content or None),
        source_url=(results["source_url"].content or None),
        had_secrets=had_secrets,
        per_field_summary=summaries,
        badge=badge,
        auto_extract_graph=auto_extract,
    )
