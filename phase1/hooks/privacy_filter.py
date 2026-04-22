"""
Privacy Filter — Edge-layer content sanitisation
==================================================

Concept inspired by claude-mem's dual-tag privacy system (thedotmack/claude-mem,
AGPL-3.0), reimplemented from scratch.

The idea: wrap anything you don't want persisted in  <private>...</private>
tags. This filter is called on the `memory` endpoint BEFORE the content
reaches Chroma / SQLite — so even if the pipeline later logs content for
debugging, the secret is already gone.

Additionally, a defensive regex pass redacts patterns that LOOK like secrets
even without tags. This is a safety net, not a guarantee: never rely on
pattern-based redaction alone for high-value credentials.

Behaviour
---------
    Input:  "Here is my plan. <private>API_KEY=sk_live_abc123</private> and ..."
    Output: "Here is my plan.  and ..."
    Side-effect: logs a WARNING with fingerprint (first 4 chars of SHA-256).

    Input:  "AWS key is AKIAIOSFODNN7EXAMPLE"
    Output: "AWS key is [REDACTED:aws_key]"
    Side-effect: WARNING log.

Two-tier design
---------------
    Tier 1 (hard strip):     <private>...</private>, <secret>...</secret>
    Tier 2 (pattern redact): known credential shapes
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Pattern

log = logging.getLogger("supermemory.privacy")


# --------------------------------------------------------------------------
#  Pattern registry
# --------------------------------------------------------------------------

@dataclass
class SecretPattern:
    name: str
    regex: Pattern
    description: str


# Patterns are deliberately conservative — we prefer false negatives over
# false positives (don't redact legitimate content by mistake).
_PATTERNS: list[SecretPattern] = [
    SecretPattern(
        name="aws_access_key",
        regex=re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        description="AWS access key ID",
    ),
    SecretPattern(
        name="stripe_live_key",
        regex=re.compile(r"\bsk_live_[0-9a-zA-Z]{24,}\b"),
        description="Stripe live secret key",
    ),
    SecretPattern(
        name="stripe_test_key",
        regex=re.compile(r"\bsk_test_[0-9a-zA-Z]{24,}\b"),
        description="Stripe test secret key",
    ),
    SecretPattern(
        name="github_pat",
        regex=re.compile(r"\bghp_[0-9a-zA-Z]{36}\b"),
        description="GitHub personal access token",
    ),
    SecretPattern(
        name="github_oauth",
        regex=re.compile(r"\bgho_[0-9a-zA-Z]{36}\b"),
        description="GitHub OAuth token",
    ),
    SecretPattern(
        name="openai_key",
        regex=re.compile(r"\bsk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b"),
        description="OpenAI API key (legacy format)",
    ),
    SecretPattern(
        name="anthropic_key",
        regex=re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{90,}\b"),
        description="Anthropic API key",
    ),
    SecretPattern(
        name="jwt",
        regex=re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
        description="JSON Web Token",
    ),
    SecretPattern(
        name="private_key_pem",
        regex=re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
            r"[\s\S]+?"
            r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
            re.MULTILINE,
        ),
        description="PEM private key block",
    ),
    # NOTE: deliberately NOT redacting generic "password: ..." — too many
    # false positives (docs, error messages, etc.). Users should use
    # <private> tags for structured credentials.
]

# Tag stripping — greedy-but-bounded, case-insensitive
_PRIVATE_TAG_RE = re.compile(r"<(private|secret)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_TAG_RE = re.compile(r"<(private|secret)\b[^>]*>", re.IGNORECASE)


# --------------------------------------------------------------------------
#  Filter result
# --------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Rich result so the caller can decide whether to accept/reject."""
    content: str
    original_length: int
    final_length: int
    private_blocks_stripped: int = 0
    patterns_redacted: list[str] = field(default_factory=list)
    unclosed_tags_found: bool = False
    rejected: bool = False
    rejection_reason: str = ""

    @property
    def had_secrets(self) -> bool:
        return self.private_blocks_stripped > 0 or bool(self.patterns_redacted)

    def summary(self) -> str:
        if not self.had_secrets and not self.unclosed_tags_found:
            return "clean"
        bits = []
        if self.private_blocks_stripped:
            bits.append(f"{self.private_blocks_stripped} private block(s) stripped")
        if self.patterns_redacted:
            bits.append(f"patterns redacted: {', '.join(self.patterns_redacted)}")
        if self.unclosed_tags_found:
            bits.append("UNCLOSED private/secret tag detected")
        return "; ".join(bits)


# --------------------------------------------------------------------------
#  Core filter
# --------------------------------------------------------------------------

def _fingerprint(s: str) -> str:
    """Short, non-reversible fingerprint for log auditing."""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:8]


def filter_content(
    content: str,
    *,
    strict_unclosed: bool = True,
    custom_patterns: list[SecretPattern] | None = None,
    enabled: bool = True,
    strip_private_tags: bool = True,
    enabled_patterns: dict[str, bool] | None = None,
) -> FilterResult:
    """Sanitise content before persistence.

    Parameters
    ----------
    content : the raw memory content submitted by the user.
    strict_unclosed : if True, reject content containing unclosed
        <private> / <secret> tags. An unclosed tag often means the user
        intended to wrap a secret but made a typo — better to refuse than
        to silently store it.
    custom_patterns : additional patterns to check (appended to built-ins).
    enabled : master toggle. When False the function short-circuits and
        returns the content unchanged (dashboard-driven).
    strip_private_tags : when False, tier-1 <private>/<secret> stripping
        is skipped. Unclosed-tag detection also skipped in that case.
    enabled_patterns : optional `{pattern_name: bool}` map. Missing names
        default to True (enabled). Disabling a name skips that built-in
        pattern. `custom_patterns` are unaffected by this map.

    Returns
    -------
    FilterResult with sanitised content and audit info.
    """
    if not content:
        return FilterResult(content="", original_length=0, final_length=0)

    original_length = len(content)

    # Master toggle — short-circuit, content passes through untouched.
    if not enabled:
        return FilterResult(
            content=content,
            original_length=original_length,
            final_length=original_length,
        )

    # ---- Tier 1: strip complete <private>...</private> and <secret>...</secret>
    private_count = 0
    sanitised = content
    unclosed_found = False
    if strip_private_tags:
        for m in _PRIVATE_TAG_RE.finditer(content):
            private_count += 1
            inner = m.group(0)
            log.warning(
                "Stripped <%s> block (fp=%s, len=%d)",
                m.group(1).lower(), _fingerprint(inner), len(inner),
            )
        sanitised = _PRIVATE_TAG_RE.sub("", content)

        # ---- Check for UNCLOSED tags after stripping closed ones
        unclosed_found = bool(_UNCLOSED_TAG_RE.search(sanitised))

    # ---- Tier 2: pattern redaction
    redacted_patterns: list[str] = []
    _toggles = enabled_patterns or {}
    patterns = [p for p in _PATTERNS if _toggles.get(p.name, True)]
    if custom_patterns:
        patterns.extend(custom_patterns)

    for p in patterns:
        def _repl(match):
            log.warning(
                "Redacted %s pattern (fp=%s)",
                p.name, _fingerprint(match.group(0)),
            )
            return f"[REDACTED:{p.name}]"

        new_sanitised, n_subs = p.regex.subn(_repl, sanitised)
        if n_subs:
            redacted_patterns.append(f"{p.name}×{n_subs}")
            sanitised = new_sanitised

    # ---- Decide on rejection
    rejected = False
    rejection_reason = ""
    if unclosed_found and strict_unclosed:
        rejected = True
        rejection_reason = (
            "Content contains an unclosed <private> or <secret> tag. "
            "This looks like a typo; refusing to store to prevent accidental "
            "leakage. Fix the tag and resubmit."
        )

    result = FilterResult(
        content=sanitised,
        original_length=original_length,
        final_length=len(sanitised),
        private_blocks_stripped=private_count,
        patterns_redacted=redacted_patterns,
        unclosed_tags_found=unclosed_found,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )

    if result.had_secrets or unclosed_found:
        log.info("privacy_filter: %s", result.summary())

    return result


# --------------------------------------------------------------------------
#  FastMCP integration helper
# --------------------------------------------------------------------------

def wrap_memory_handler(original_handler, *, strict_unclosed: bool = True):
    """Decorator that wraps your existing `memory` handler with the filter.

    Usage:

        from hooks.privacy_filter import wrap_memory_handler

        @mcp.tool()
        @wrap_memory_handler
        def memory(content: str, project: str = "default", ...):
            # your existing logic
            ...

    Or if your handler isn't a direct function, patch manually:

        filtered = filter_content(raw_content)
        if filtered.rejected:
            return {"ok": False, "error": filtered.rejection_reason}
        store_memory(filtered.content, ...)
    """
    import functools

    @functools.wraps(original_handler)
    def wrapper(content: str, *args, **kwargs):
        filtered = filter_content(content, strict_unclosed=strict_unclosed)
        if filtered.rejected:
            return {
                "ok": False,
                "error": filtered.rejection_reason,
                "audit": filtered.summary(),
            }
        # Call through with sanitised content
        result = original_handler(filtered.content, *args, **kwargs)
        if isinstance(result, dict):
            result.setdefault("privacy", {})
            result["privacy"]["summary"] = filtered.summary()
            result["privacy"]["secrets_found"] = filtered.had_secrets
        return result

    return wrapper
