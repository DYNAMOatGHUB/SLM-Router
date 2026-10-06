"""NFR-7: model text must never become Python code."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "tinyagent"
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "literal_eval"}
FORBIDDEN_IMPORTS = {"pickle", "importlib"}


class StaticGateTest(unittest.TestCase):
    def test_no_dynamic_execution(self):
        findings = []
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                        findings.append(f"{path}:{node.lineno} call {node.func.id}")
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "literal_eval":
                        findings.append(f"{path}:{node.lineno} literal_eval")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".", 1)[0] in FORBIDDEN_IMPORTS:
                            findings.append(f"{path}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".", 1)[0] in FORBIDDEN_IMPORTS:
                        findings.append(f"{path}:{node.lineno} import {node.module}")
        self.assertEqual(findings, [])
