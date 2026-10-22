"""Synthetic, offline regression tests. Never uses a real Discord token or guild."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_storage.py"


class HQPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / "hq_layouts.json"
        self.file.write_text(json.dumps({"fixture-guild": {
            "claims": {"A1": {"name": "PRIVATE FIXTURE PLAYER"}},
            "cal_points": {}
        }}), encoding="utf-8")
        self.env = {
            **os.environ,
            "DISCORD_BOT_TOKEN": "synthetic-test-token-do-not-use",
            "DATA_FILE": str(self.file),
            "RAILWAY_VOLUME_MOUNT_PATH": str(self.root)
        }

    def run_preflight(self, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), "--require-volume"],
                              env=env or self.env, capture_output=True, text=True)

    def test_valid_file(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1 guild layout(s), 1 saved assignment(s)", result.stdout)
        self.assertNotIn("PRIVATE FIXTURE PLAYER", result.stdout)
        self.assertNotIn("fixture-guild", result.stdout)
        self.assertNotIn("synthetic-test-token", result.stdout)

    def test_missing_file_fails(self):
        self.file.unlink()
        self.assertEqual(self.run_preflight().returncode, 4)

    def test_malformed_file_fails(self):
        self.file.write_text("{invalid")
        self.assertEqual(self.run_preflight().returncode, 5)

    def test_empty_file_fails(self):
        self.file.write_text("{}")
        self.assertEqual(self.run_preflight().returncode, 6)

    def test_missing_volume_fails(self):
        env = {**self.env}
        env.pop("RAILWAY_VOLUME_MOUNT_PATH")
        self.assertEqual(self.run_preflight(env).returncode, 3)

    def test_data_file_outside_volume_fails(self):
        env = {**self.env, "RAILWAY_VOLUME_MOUNT_PATH": str(self.root / "elsewhere")}
        self.assertEqual(self.run_preflight(env).returncode, 3)

    def test_inspect_only_without_token_preserves_file(self):
        env = {**self.env}
        env.pop("DISCORD_BOT_TOKEN", None)
        before = self.file.read_bytes()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--require-volume", "--inspect-only"],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("HQ preflight: INSPECT ONLY", result.stdout)
        self.assertIn("INSPECTION COMPLETE", result.stdout)
        self.assertIn("SHA-256:", result.stdout)
        self.assertNotIn("PRIVATE FIXTURE PLAYER", result.stdout)
        self.assertNotIn("fixture-guild", result.stdout)
        self.assertEqual(self.file.read_bytes(), before, "Inspection must not change saved layouts")

    def test_inspect_only_missing_volume_still_fails(self):
        env = {**self.env}
        env.pop("DISCORD_BOT_TOKEN", None)
        env.pop("RAILWAY_VOLUME_MOUNT_PATH", None)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--require-volume", "--inspect-only"],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertIn("no Railway volume mount", result.stdout)

    def test_inspect_only_missing_file_still_fails(self):
        env = {**self.env}
        env.pop("DISCORD_BOT_TOKEN", None)
        self.file.unlink()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--require-volume", "--inspect-only"],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 4)

    def test_inspect_only_rejects_malformed_layout(self):
        env = {**self.env}
        env.pop("DISCORD_BOT_TOKEN", None)
        self.file.write_text('{"guild":{"notClaims":{}}}', encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--require-volume", "--inspect-only"],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 6)

    def test_missing_bot_token_fails(self):
        env = {**self.env}
        env.pop("DISCORD_BOT_TOKEN")
        result = self.run_preflight(env)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("synthetic-test-token", result.stdout)


if __name__ == "__main__":
    unittest.main()
