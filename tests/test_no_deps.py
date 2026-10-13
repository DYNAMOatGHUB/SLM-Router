"""NFR-1: runtime package imports only standard-library modules."""

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "slmrouter"


class NoDependenciesTest(unittest.TestCase):
    def test_only_stdlib_or_slmrouter(self):
        disallowed = []
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level:
                    continue
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    root = name.split(".", 1)[0]
                    if root not in sys.stdlib_module_names | {"slmrouter"}:
                        disallowed.append(f"{path}: {name}")
        self.assertEqual(disallowed, [])
