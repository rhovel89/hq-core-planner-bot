"""Offline atomic-save tests extracted from the real bot source; no Discord login."""
import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "hq_core_planner_bot.py"


def persistence_namespace(destination):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"save_data", "load_data"}]
    if len(functions) != 2:
        raise AssertionError("Expected both original bot persistence functions")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    ns = {"os": os, "json": json, "shutil": shutil, "tempfile": tempfile,
          "Dict": Dict, "Any": Any, "DATA_FILE": str(destination)}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns


class AtomicPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_file = self.root / "hq_layouts.json"
        self.ns = persistence_namespace(self.data_file)

    def test_first_save_writes_valid_json(self):
        self.ns["save_data"]({"synthetic": {"claims": {"E6": {"user_name": "Fixture A"}}}})
        self.assertEqual(self.ns["load_data"]()["synthetic"]["claims"]["E6"]["user_name"], "Fixture A")
        self.assertFalse((self.root / "hq_layouts.json.bak").exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_second_save_preserves_previous_generation(self):
        first = {"synthetic": {"claims": {"E6": {"user_name": "Fixture A"}}}}
        second = {"synthetic": {"claims": {"G7": {"user_name": "Fixture B"}}}}
        self.ns["save_data"](first)
        self.ns["save_data"](second)
        self.assertEqual(json.loads(self.data_file.read_text()), second)
        self.assertEqual(json.loads((self.root / "hq_layouts.json.bak").read_text()), first)

    def test_invalid_json_payload_never_truncates_live_file(self):
        self.ns["save_data"]({"synthetic": {"claims": {}}})
        original = self.data_file.read_bytes()
        with self.assertRaises(TypeError):
            self.ns["save_data"]({"broken": object()})
        self.assertEqual(self.data_file.read_bytes(), original)

    def test_failed_backup_copy_preserves_live_file(self):
        self.ns["save_data"]({"synthetic": {"claims": {"E6": {}}}})
        old = self.data_file.read_bytes()
        with patch.object(shutil, "copyfileobj", side_effect=OSError("synthetic backup failure")):
            with self.assertRaises(OSError):
                self.ns["save_data"]({"synthetic": {"claims": {"G7": {}}}})
        self.assertEqual(self.data_file.read_bytes(), old)
        self.assertFalse(list(self.root.glob(".hq-*.tmp")))


if __name__ == "__main__":
    unittest.main()
