"""Exercise original bot start/ready log contracts without a Discord connection."""
import ast
import asyncio
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE=Path(__file__).resolve().parents[1]/"hq_core_planner_bot.py"

def load_method(name, guild_id=None):
    tree=ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    classes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="HQCorePlanner"]
    assert len(classes)==1
    functions=[x for x in classes[0].body if isinstance(x,ast.AsyncFunctionDef) and x.name==name]
    assert len(functions)==1
    function=functions[0]
    function.decorator_list=[]
    compiled=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(SOURCE),"exec")
    namespace={"layout_group":object(),"slot_group":object(),"cal_group":object(),
      "GUILD_ID_ENV":guild_id, "discord":SimpleNamespace(Object=lambda id:SimpleNamespace(id=id))}
    exec(compiled,namespace)
    return namespace[name]

class MockTree:
    def __init__(self, fail=False):
        self.commands=[]
        self.guild=None
        self.fail=fail
    def add_command(self,item):
        self.commands.append(item)
    def copy_global_to(self,guild=None):
        self.guild=guild
    async def sync(self,guild=None):
        if self.fail:raise OSError("Synthetic gateway/sync failure")
        self.guild=guild
        return self.commands

class StartDiagnosticsTests(unittest.TestCase):
    def test_global_sync_and_gateway_ready(self):
        worker=SimpleNamespace(tree=MockTree(),application_id=123456789,guilds=[object(),object()])
        output=StringIO()
        with redirect_stdout(output):
            asyncio.run(load_method("setup_hook")(worker))
            asyncio.run(load_method("on_ready")(worker))
        log=output.getvalue()
        self.assertIn("HQ_SYNC_START scope=global",log)
        self.assertIn("HQ_SYNC_OK scope=global commands=3",log)
        self.assertIn("HQ_BOT_READY app_id=123456789 guilds=2",log)
        self.assertEqual(len(worker.tree.commands),3)
        self.assertNotIn("DISCORD_BOT_TOKEN",log)

    def test_guild_scoped_sync_no_member_detail(self):
        worker=SimpleNamespace(tree=MockTree())
        output=StringIO()
        with redirect_stdout(output):
            asyncio.run(load_method("setup_hook",guild_id="789")(worker))
        log=output.getvalue()
        self.assertIn("HQ_SYNC_START scope=guild",log)
        self.assertIn("HQ_SYNC_OK scope=guild commands=3",log)
        self.assertEqual(worker.tree.guild.id,789)

    def test_sync_failure_is_visible_without_exception_details(self):
        worker=SimpleNamespace(tree=MockTree(fail=True))
        output=StringIO()
        with redirect_stdout(output):
            with self.assertRaises(OSError):
                asyncio.run(load_method("setup_hook")(worker))
        self.assertIn("HQ_SYNC_FAILED type=OSError",output.getvalue())
        self.assertNotIn("Synthetic gateway/sync failure",output.getvalue())

if __name__=="__main__":
    unittest.main()
