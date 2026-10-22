"""Offline response-contract test for original /layout storage; no Discord or Railway connection."""
import ast
import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SOURCE=Path(__file__).resolve().parents[1]/"hq_core_planner_bot.py"

def load_storage_handler(data_file):
    tree=ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    methods=[node for node in tree.body if isinstance(node,ast.AsyncFunctionDef) and node.name=="layout_storage"]
    if len(methods)!=1:
        raise AssertionError("Expected precisely one original /layout storage handler")
    fn=methods[0]
    fn.decorator_list=[]
    code=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    ns={"os":os,"DATA_FILE":str(data_file),"discord":SimpleNamespace(Interaction=object)}
    exec(compile(code,str(SOURCE),"exec"),ns)
    return ns["layout_storage"]

class Recorder:
    def __init__(self):self.messages=[]
    async def send_message(self,text,**kwargs):self.messages.append((text,kwargs))

class StorageReplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"hq_layouts.json"

    def invoke(self):
        response=Recorder()
        async def run():
            await asyncio.wait_for(
                load_storage_handler(self.path)(SimpleNamespace(response=response)),timeout=1
            )
        asyncio.run(run())
        self.assertEqual(len(response.messages),1)
        self.assertIs(response.messages[0][1].get("ephemeral"),True)
        return response.messages[0][0]

    def test_absent_file_responds_quickly_and_privately(self):
        reply=self.invoke()
        self.assertIn("EXISTS = \`False\`",reply)
        self.assertIn("SIZE = \`0\` bytes",reply)
        self.assertIn(str(self.path),reply)

    def test_present_file_responds_quickly_and_privately(self):
        self.path.write_text('{"synthetic":true}',encoding="utf-8")
        reply=self.invoke()
        self.assertIn("EXISTS = \`True\`",reply)
        self.assertIn("SIZE = \`18\` bytes",reply)
        self.assertEqual(self.path.read_text(), '{"synthetic":true}')

if __name__=="__main__":
    unittest.main()
