"""Tests for the dashboard subpackage.

Covers: config load/save/hot-reload, policy orchestration (blocklist,
max-chars, pattern toggles, master disable), privacy filter toggle kwargs.

Run:
    python phase1/tests/test_dashboard.py
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Make phase1 importable when run from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase1.dashboard.config import Config, _DEFAULTS, _deep_merge
from phase1.dashboard.integration import apply_save_policy
from phase1.dashboard import config as config_mod
from phase1.dashboard import audit as audit_mod
from phase1.hooks.privacy_filter import filter_content


# ── Config ─────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "config.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_defaults_written_on_first_access(self):
        c = Config(self.path)
        self.assertTrue(self.path.exists())
        d = c.get()
        self.assertTrue(d["privacy"]["enabled"])
        self.assertTrue(d["privacy"]["patterns"]["aws_access_key"])

    def test_deep_merge_preserves_user_values(self):
        # Pre-write a partial config: user disabled jwt only
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "privacy": {"patterns": {"jwt": False}},
        }))
        c = Config(self.path)
        d = c.get()
        self.assertFalse(d["privacy"]["patterns"]["jwt"])
        # Defaults for un-specified keys
        self.assertTrue(d["privacy"]["patterns"]["aws_access_key"])
        self.assertTrue(d["privacy"]["enabled"])

    def test_save_and_reload(self):
        c = Config(self.path)
        c.save({"capture": {"max_content_chars": 1234}})
        c2 = Config(self.path)
        self.assertEqual(c2.get()["capture"]["max_content_chars"], 1234)

    def test_hot_reload_picks_up_external_changes(self):
        c = Config(self.path)
        self.assertEqual(c.get()["capture"]["max_content_chars"], 200000)
        # Simulate external edit
        time.sleep(0.01)  # ensure mtime differs on fast filesystems
        self.path.write_text(json.dumps({
            "capture": {"max_content_chars": 42},
        }))
        # Force mtime advance on systems with 1s resolution
        new_mtime = time.time() + 1
        os.utime(self.path, (new_mtime, new_mtime))
        self.assertEqual(c.get()["capture"]["max_content_chars"], 42)

    def test_corrupt_file_does_not_overwrite(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json")
        c = Config(self.path)
        # Falls back to defaults in-memory but doesn't touch the file
        d = c.get()
        self.assertEqual(d["capture"]["max_content_chars"],
                         _DEFAULTS["capture"]["max_content_chars"])

    def test_deep_merge_adds_new_default_keys(self):
        out = _deep_merge(
            {"a": {"x": 1, "y": 2}, "b": True},
            {"a": {"x": 99}},
        )
        self.assertEqual(out, {"a": {"x": 99, "y": 2}, "b": True})


# ── Integration (policy) ───────────────────────────────────────────

class TestPolicy(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg_path = Path(self.tmp) / "config.json"
        self.audit_path = Path(self.tmp) / "audit.jsonl"
        # Point singletons at temp files
        config_mod._singleton = Config(self.cfg_path)
        audit_mod.set_path_for_tests(self.audit_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        audit_mod.set_path_for_tests(None)
        config_mod._singleton = None

    def _set(self, patch: dict):
        config_mod._singleton.save(patch)

    def test_clean_passthrough(self):
        r = apply_save_policy(content="hello world", project="default")
        self.assertFalse(r.rejected)
        self.assertFalse(r.had_secrets)
        self.assertEqual(r.content, "hello world")
        self.assertEqual(r.badge, "")
        self.assertTrue(r.auto_extract_graph)

    def test_blocked_project_rejected(self):
        self._set({"capture": {"blocked_projects": ["secret-ops"]}})
        r = apply_save_policy(content="x", project="secret-ops")
        self.assertTrue(r.rejected)
        self.assertIn("secret-ops", r.rejection_reason)
        self.assertIn("blocked", r.rejection_reason)

    def test_max_content_chars_rejected(self):
        self._set({"capture": {"max_content_chars": 10}})
        r = apply_save_policy(content="a" * 50, project="default")
        self.assertTrue(r.rejected)
        self.assertIn("50", r.rejection_reason)
        self.assertIn("10", r.rejection_reason)

    def test_master_disable_bypasses_filter(self):
        self._set({"privacy": {"enabled": False}})
        r = apply_save_policy(
            content="AKIAIOSFODNN7EXAMPLE should leak",
            project="default",
        )
        self.assertFalse(r.rejected)
        # Filter disabled → no redaction
        self.assertIn("AKIAIOSFODNN7EXAMPLE", r.content)
        self.assertFalse(r.had_secrets)

    def test_pattern_toggle_skips_disabled_pattern(self):
        self._set({"privacy": {"patterns": {"aws_access_key": False}}})
        r = apply_save_policy(
            content="AKIAIOSFODNN7EXAMPLE ok",
            project="default",
        )
        self.assertFalse(r.rejected)
        self.assertIn("AKIAIOSFODNN7EXAMPLE", r.content)
        self.assertFalse(r.had_secrets)

    def test_pattern_toggle_keeps_enabled_patterns(self):
        # aws off, github on — github should still redact
        self._set({"privacy": {"patterns": {"aws_access_key": False}}})
        r = apply_save_policy(
            content="aws AKIAIOSFODNN7EXAMPLE gh ghp_" + "A"*36,
            project="default",
        )
        self.assertIn("AKIAIOSFODNN7EXAMPLE", r.content)
        self.assertIn("[REDACTED:github_pat]", r.content)

    def test_strip_tags_toggle(self):
        self._set({"privacy": {"strip_private_tags": False}})
        r = apply_save_policy(
            content="keep <private>it</private>",
            project="default",
        )
        self.assertFalse(r.rejected)
        self.assertIn("<private>it</private>", r.content)

    def test_strict_unclosed_off_allows_save(self):
        self._set({"privacy": {"strict_unclosed_tags": False}})
        r = apply_save_policy(
            content="open <private>oops",
            project="default",
        )
        self.assertFalse(r.rejected)

    def test_metadata_fields_filtered(self):
        r = apply_save_policy(
            content="safe",
            project="default",
            title="stripe sk_live_" + "a"*30,
            description="<private>X</private> hidden",
            source_url="https://x/AKIAIOSFODNN7EXAMPLE/y",
        )
        self.assertFalse(r.rejected)
        self.assertIn("[REDACTED:stripe_live_key]", r.title)
        self.assertNotIn("<private>", r.description)
        self.assertIn("[REDACTED:aws_access_key]", r.source_url)
        # All three fields contributed to summary
        self.assertIn("title", r.per_field_summary)
        self.assertIn("description", r.per_field_summary)
        self.assertIn("source_url", r.per_field_summary)

    def test_auto_extract_from_config(self):
        self._set({"capture": {"auto_extract_graph": False}})
        r = apply_save_policy(content="hi", project="default")
        self.assertFalse(r.auto_extract_graph)

    def test_audit_log_records_save(self):
        apply_save_policy(content="hi", project="default")
        entries = audit_mod.recent(5)
        self.assertTrue(any(e.get("event") == "save_ok" for e in entries))


# ── filter_content kwarg behaviour ─────────────────────────────────

class TestFilterToggleKwargs(unittest.TestCase):

    def test_enabled_false_skips_all(self):
        r = filter_content(
            "AKIAIOSFODNN7EXAMPLE <private>x</private>",
            enabled=False,
        )
        self.assertFalse(r.rejected)
        self.assertFalse(r.had_secrets)
        self.assertIn("AKIAIOSFODNN7EXAMPLE", r.content)
        self.assertIn("<private>", r.content)

    def test_strip_tags_false_keeps_private_block(self):
        r = filter_content("<private>x</private>", strip_private_tags=False)
        self.assertEqual(r.content, "<private>x</private>")
        self.assertEqual(r.private_blocks_stripped, 0)

    def test_strip_tags_false_disables_unclosed_rejection(self):
        r = filter_content("<private>no end", strip_private_tags=True)
        self.assertTrue(r.rejected)  # control: strict+strip → rejected
        r2 = filter_content("<private>no end", strip_private_tags=False)
        self.assertFalse(r2.rejected)

    def test_enabled_patterns_disables_specific_built_in(self):
        r = filter_content(
            "AKIAIOSFODNN7EXAMPLE",
            enabled_patterns={"aws_access_key": False},
        )
        self.assertIn("AKIAIOSFODNN7EXAMPLE", r.content)

    def test_enabled_patterns_missing_keys_default_true(self):
        r = filter_content(
            "AKIAIOSFODNN7EXAMPLE",
            enabled_patterns={"github_pat": False},  # aws unspecified → on
        )
        self.assertIn("[REDACTED:aws_access_key]", r.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
