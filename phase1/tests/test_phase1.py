"""
Tests for Phase 1 — Progressive Disclosure + Privacy Filter
============================================================

Run with:
    cd /Applications/ServBay/www/supermemory.granaria/phase1
    python -m pytest tests/ -v

Or (without pytest):
    python tests/test_phase1.py
"""

import sys
import os
import unittest
from pathlib import Path

# Make imports work whether run from phase1/ or tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from hooks.privacy_filter import (
    filter_content,
    FilterResult,
    SecretPattern,
)
from tools.recall_progressive import (
    IndexHit,
    FullMemory,
    _derive_title,
    _safe_meta,
    build_index,
    fetch_by_ids,
)
import re


# --------------------------------------------------------------------------
#  Privacy filter tests
# --------------------------------------------------------------------------

class TestPrivacyFilter(unittest.TestCase):

    def test_clean_content_passes_through(self):
        r = filter_content("This is ordinary text with no secrets.")
        self.assertFalse(r.rejected)
        self.assertFalse(r.had_secrets)
        self.assertEqual(r.content, "This is ordinary text with no secrets.")

    def test_empty_content(self):
        r = filter_content("")
        self.assertEqual(r.final_length, 0)
        self.assertFalse(r.rejected)

    def test_private_tag_stripped(self):
        raw = "Plan: deploy. <private>KEY=abc123def</private> Done."
        r = filter_content(raw)
        self.assertNotIn("abc123def", r.content)
        self.assertNotIn("<private>", r.content)
        self.assertEqual(r.private_blocks_stripped, 1)
        self.assertFalse(r.rejected)

    def test_multiple_private_tags(self):
        raw = "<private>a</private> middle <private>b</private> end"
        r = filter_content(raw)
        self.assertEqual(r.private_blocks_stripped, 2)
        self.assertNotIn("a</", r.content)
        self.assertNotIn("b</", r.content)

    def test_secret_tag_variant(self):
        r = filter_content("ok <secret>hush</secret> done")
        self.assertNotIn("hush", r.content)
        self.assertEqual(r.private_blocks_stripped, 1)

    def test_case_insensitive_tags(self):
        r = filter_content("<PRIVATE>x</PRIVATE> and <Private>y</Private>")
        self.assertEqual(r.private_blocks_stripped, 2)
        self.assertNotIn("x", r.content.replace("and", ""))  # rough check

    def test_multiline_private_block(self):
        raw = "start\n<private>\nmulti\nline\nsecret\n</private>\nend"
        r = filter_content(raw)
        self.assertNotIn("multi", r.content)
        self.assertNotIn("line", r.content)
        self.assertIn("start", r.content)
        self.assertIn("end", r.content)

    def test_unclosed_tag_rejected_strict(self):
        raw = "start <private>oops forgot closing tag"
        r = filter_content(raw, strict_unclosed=True)
        self.assertTrue(r.rejected)
        self.assertIn("unclosed", r.rejection_reason.lower())

    def test_unclosed_tag_allowed_non_strict(self):
        raw = "start <private>oops"
        r = filter_content(raw, strict_unclosed=False)
        self.assertFalse(r.rejected)
        self.assertTrue(r.unclosed_tags_found)

    def test_aws_key_redacted(self):
        raw = "My key is AKIAIOSFODNN7EXAMPLE and another AKIAXXXXXXXXXXXXXXXX"
        r = filter_content(raw)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", r.content)
        self.assertIn("[REDACTED:aws_access_key]", r.content)
        self.assertTrue(any("aws_access_key" in p for p in r.patterns_redacted))

    def test_stripe_live_key_redacted(self):
        raw = "sk_live_" + "a" * 30
        r = filter_content(raw)
        self.assertNotIn("sk_live_" + "a" * 30, r.content)

    def test_github_pat_redacted(self):
        raw = "token: ghp_" + "a" * 36
        r = filter_content(raw)
        self.assertIn("[REDACTED:github_pat]", r.content)

    def test_pem_private_key_redacted(self):
        raw = (
            "Here is a key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEA...\n"
            "fakekeydata\n"
            "-----END RSA PRIVATE KEY-----\n"
            "end."
        )
        r = filter_content(raw)
        self.assertNotIn("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEA", r.content)
        self.assertIn("[REDACTED:private_key_pem]", r.content)

    def test_jwt_redacted(self):
        raw = "Authorization: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.abcdefghijXYZ"
        r = filter_content(raw)
        self.assertIn("[REDACTED:jwt]", r.content)

    def test_false_positive_resistance(self):
        # Things that should NOT be redacted
        safe_inputs = [
            "password123 is the most common password",  # word 'password'
            "AKIA prefix means AWS but this is just the letters",  # short
            "sk_test_shortkey",  # too short
            "talk about sk-ant but no real key",  # no valid format
        ]
        for text in safe_inputs:
            r = filter_content(text)
            self.assertFalse(
                r.patterns_redacted,
                f"False positive on: {text!r} → {r.patterns_redacted}",
            )

    def test_custom_pattern(self):
        custom = [
            SecretPattern(
                name="granaria_vps_token",
                regex=re.compile(r"\bGRNR-[A-F0-9]{16}\b"),
                description="Granaria VPS token",
            )
        ]
        raw = "token=GRNR-AAAA1111BBBB2222 end"
        r = filter_content(raw, custom_patterns=custom)
        self.assertIn("[REDACTED:granaria_vps_token]", r.content)


# --------------------------------------------------------------------------
#  Progressive recall helper tests (no Chroma needed)
# --------------------------------------------------------------------------

class TestRecallHelpers(unittest.TestCase):

    def test_safe_meta_none(self):
        self.assertEqual(_safe_meta(None, "x"), "")

    def test_safe_meta_missing_key(self):
        self.assertEqual(_safe_meta({"a": 1}, "b"), "")
        self.assertEqual(_safe_meta({"a": 1}, "b", default="fallback"), "fallback")

    def test_safe_meta_non_string_coerced(self):
        self.assertEqual(_safe_meta({"x": 42}, "x"), "42")

    def test_derive_title_short(self):
        self.assertEqual(_derive_title("hello world"), "hello world")

    def test_derive_title_multiline(self):
        self.assertEqual(_derive_title("first line\nsecond line\nthird"), "first line")

    def test_derive_title_truncation(self):
        long = "a" * 200
        t = _derive_title(long, max_len=80)
        self.assertEqual(len(t), 80)
        self.assertTrue(t.endswith("…"))

    def test_index_hit_serialises(self):
        h = IndexHit(
            memory_id="m1", title="t", mem_type="note",
            created_at="2026-04-22", score=0.9, project="default",
        )
        d = h.as_dict()
        self.assertEqual(d["memory_id"], "m1")
        self.assertEqual(d["score"], 0.9)


# --------------------------------------------------------------------------
#  Progressive recall integration with mock Chroma
# --------------------------------------------------------------------------

class _MockEmbedder:
    def encode(self, text):
        # Deterministic tiny embedding for tests
        return [float(hash(text) % 997) / 997.0] * 8


class _MockCollection:
    def __init__(self, data):
        """data: list of (id, doc, metadata, distance)"""
        self.data = data

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        # Ignore actual vector similarity in tests — just return top-N
        # optionally filtered by `where`.
        items = self.data
        if where:
            # Very simple where-evaluation for the fields we use
            def matches(meta):
                def check(filt):
                    for k, v in filt.items():
                        if k == "$and":
                            return all(check(sub) for sub in v)
                        if k == "$in":
                            return meta.get("_current_key") in v
                        # simple equality
                        if meta.get(k) != v:
                            return False
                    return True
                return check(where)
            items = [row for row in items if matches(row[2])]
        items = items[:n_results]
        return {
            "ids": [[row[0] for row in items]],
            "documents": [[row[1] for row in items]],
            "metadatas": [[row[2] for row in items]],
            "distances": [[row[3] for row in items]],
        }

    def get(self, *, where=None, include=None):
        items = self.data
        if where and "$and" in where:
            # Find the $in clause
            for clause in where["$and"]:
                if "memory_id" in clause and "$in" in clause["memory_id"]:
                    ids = clause["memory_id"]["$in"]
                    items = [r for r in items if r[2].get("memory_id") in ids]
                    break
        elif where and "memory_id" in where and "$in" in where["memory_id"]:
            ids = where["memory_id"]["$in"]
            items = [r for r in items if r[2].get("memory_id") in ids]
        return {
            "ids": [row[0] for row in items],
            "documents": [row[1] for row in items],
            "metadatas": [row[2] for row in items],
        }


class TestProgressiveRecall(unittest.TestCase):

    def _sample_collection(self):
        return _MockCollection([
            ("chroma-id-1", "Paarstraße 7 Energieausweis Altbau",
             {"memory_id": "m1", "title": "Energieausweis P7",
              "mem_type": "fact", "created_at": "2026-04-10", "project": "immo.granaria"},
             0.2),
            ("chroma-id-2", "VPS2 IP 187.124.248.50 root access",
             {"memory_id": "m2", "title": "VPS2 credentials",
              "mem_type": "fact", "created_at": "2026-03-15", "project": "default"},
             0.3),
            ("chroma-id-3", "GGT whitepaper section 4 tokenomics",
             {"memory_id": "m3", "title": "GGT tokenomics",
              "mem_type": "decision", "created_at": "2026-04-01", "project": "granaria-vc"},
             0.4),
        ])

    def test_build_index_returns_all_hits(self):
        coll = self._sample_collection()
        hits = build_index(coll, "Paarstraße", _MockEmbedder(), n_results=10)
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(isinstance(h, IndexHit) for h in hits))

    def test_build_index_score_is_normalised(self):
        coll = self._sample_collection()
        hits = build_index(coll, "x", _MockEmbedder())
        # score = 1 - dist, clamped 0..1
        for h in hits:
            self.assertGreaterEqual(h.score, 0.0)
            self.assertLessEqual(h.score, 1.0)

    def test_build_index_empty_query(self):
        coll = self._sample_collection()
        self.assertEqual(build_index(coll, "", _MockEmbedder()), [])
        self.assertEqual(build_index(coll, "   ", _MockEmbedder()), [])

    def test_fetch_by_ids_preserves_order(self):
        coll = self._sample_collection()
        full = fetch_by_ids(coll, ["m3", "m1"])
        self.assertEqual(len(full), 2)
        self.assertEqual(full[0].memory_id, "m3")
        self.assertEqual(full[1].memory_id, "m1")

    def test_fetch_by_ids_dedupes(self):
        coll = self._sample_collection()
        full = fetch_by_ids(coll, ["m1", "m1", "m1"])
        self.assertEqual(len(full), 1)

    def test_fetch_by_ids_empty(self):
        coll = self._sample_collection()
        self.assertEqual(fetch_by_ids(coll, []), [])

    def test_fullmemory_carries_metadata(self):
        coll = self._sample_collection()
        full = fetch_by_ids(coll, ["m1"])
        self.assertEqual(full[0].project, "immo.granaria")
        self.assertEqual(full[0].mem_type, "fact")
        self.assertIn("memory_id", full[0].metadata)

    def test_fetch_by_ids_dedupes_multi_chunk_memory(self):
        """A memory split into N chunks must return ONE FullMemory, not N."""
        coll = _MockCollection([
            ("mid-x#c000", "Part one of the text. ",
             {"memory_id": "mid-x", "chunk_index": 0, "chunk_count": 3,
              "title": "", "project": "default"}, 0.1),
            ("mid-x#c001", "Part two of the text. ",
             {"memory_id": "mid-x", "chunk_index": 1, "chunk_count": 3,
              "title": "", "project": "default"}, 0.2),
            ("mid-x#c002", "Part three of the text.",
             {"memory_id": "mid-x", "chunk_index": 2, "chunk_count": 3,
              "title": "", "project": "default"}, 0.3),
        ])
        full = fetch_by_ids(coll, ["mid-x"])
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].memory_id, "mid-x")
        # Default behaviour: chunk 0 content (best-effort, lossy)
        self.assertEqual(full[0].content, "Part one of the text. ")

    def test_fetch_by_ids_uses_content_loader(self):
        """content_loader overrides chunk-based content reconstruction."""
        coll = _MockCollection([
            ("mid-y#c000", "chunk zero",
             {"memory_id": "mid-y", "chunk_index": 0, "chunk_count": 2,
              "project": "default"}, 0.1),
            ("mid-y#c001", "chunk one",
             {"memory_id": "mid-y", "chunk_index": 1, "chunk_count": 2,
              "project": "default"}, 0.2),
        ])
        loaded = {"mid-y": "the unsplit authoritative Volltext"}
        full = fetch_by_ids(coll, ["mid-y"], content_loader=lambda m: loaded.get(m))
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].content, "the unsplit authoritative Volltext")

    def test_fetch_by_ids_content_loader_failure_falls_back(self):
        """If the loader raises, fall back to chunk content — never crash."""
        coll = _MockCollection([
            ("mid-z#c000", "backup chunk content",
             {"memory_id": "mid-z", "chunk_index": 0, "chunk_count": 1,
              "project": "default"}, 0.1),
        ])
        def bad_loader(_):
            raise RuntimeError("simulated db outage")
        full = fetch_by_ids(coll, ["mid-z"], content_loader=bad_loader)
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].content, "backup chunk content")


# --------------------------------------------------------------------------
#  Token-saving sanity check
# --------------------------------------------------------------------------

class TestTokenEconomics(unittest.TestCase):
    """Soft assertion that the index is dramatically smaller than full payload."""

    def test_index_payload_smaller_than_full(self):
        # Typical Granaria memory: 400–2000 chars
        full_content = "x" * 1500
        meta = {
            "memory_id": "m-test",
            "title": "Test memory with reasonable title length",
            "mem_type": "fact",
            "created_at": "2026-04-22T10:00:00Z",
            "project": "default",
        }
        hit = IndexHit(
            memory_id=meta["memory_id"],
            title=meta["title"],
            mem_type=meta["mem_type"],
            created_at=meta["created_at"],
            score=0.87,
            project=meta["project"],
        )
        full = FullMemory(
            memory_id=meta["memory_id"],
            content=full_content,
            title=meta["title"],
            mem_type=meta["mem_type"],
            created_at=meta["created_at"],
            project=meta["project"],
            metadata=meta,
        )
        import json
        index_bytes = len(json.dumps(hit.as_dict()))
        full_bytes = len(json.dumps(full.as_dict()))
        ratio = full_bytes / index_bytes
        # We want at least 5× smaller — typically 8–10×
        self.assertGreater(
            ratio, 5.0,
            f"Index not small enough: {index_bytes} vs {full_bytes} (ratio {ratio:.1f}×)",
        )


if __name__ == "__main__":
    # Run without pytest
    unittest.main(verbosity=2)
