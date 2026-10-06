import unittest
from typing import Optional
from tinyagent.reflection import tiny_tool
from tinyagent.registry import ToolRegistry

class TestReflection(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def test_reflection_basic(self):
        @tiny_tool(registry=self.registry)
        def f(city: str, unit: str = "c") -> str:
            pass
        
        spec = self.registry.get("f")
        self.assertIsNotNone(spec)
        self.assertEqual(len(spec.params), 2)
        
        p1, p2 = spec.params
        self.assertEqual(p1.name, "city")
        self.assertEqual(p1.type, "str")
        self.assertTrue(p1.required)
        self.assertFalse(p1.optional)
        
        self.assertEqual(p2.name, "unit")
        self.assertEqual(p2.type, "str")
        self.assertFalse(p2.required)
        self.assertEqual(p2.default, "c")
        self.assertFalse(p2.optional)

    def test_reflection_types(self):
        @tiny_tool(registry=self.registry)
        def f(a: int, b: float, flag: bool = False):
            pass
        
        spec = self.registry.get("f")
        p1, p2, p3 = spec.params
        
        self.assertEqual(p1.type, "int")
        self.assertTrue(p1.required)
        
        self.assertEqual(p2.type, "float")
        self.assertTrue(p2.required)
        
        self.assertEqual(p3.type, "bool")
        self.assertFalse(p3.required)
        self.assertEqual(p3.default, False)

    def test_reflection_missing_annotation(self):
        @tiny_tool(registry=self.registry)
        def f(x):
            pass
            
        spec = self.registry.get("f")
        p1 = spec.params[0]
        self.assertEqual(p1.name, "x")
        self.assertEqual(p1.type, "str")
        self.assertTrue(p1.required)

    def test_reflection_optional(self):
        @tiny_tool(registry=self.registry)
        def f(n: Optional[int] = None):
            pass
            
        spec = self.registry.get("f")
        p1 = spec.params[0]
        self.assertEqual(p1.type, "int")
        self.assertTrue(p1.optional)
        self.assertFalse(p1.required)
        self.assertIsNone(p1.default)

    def test_unsupported_type(self):
        with self.assertRaisesRegex(TypeError, "unsupported type"):
            @tiny_tool(registry=self.registry)
            def f(items: list):
                pass

    def test_unsupported_varargs(self):
        with self.assertRaisesRegex(TypeError, "unsupported kind"):
            @tiny_tool(registry=self.registry)
            def f(*args):
                pass

    def test_unsupported_positional_only(self):
        with self.assertRaisesRegex(TypeError, "unsupported kind"):
            @tiny_tool(registry=self.registry)
            def f(a, /, b):
                pass

    def test_docstring_extraction(self):
        @tiny_tool(registry=self.registry)
        def f():
            """
            Get weather.
            More text.
            """
            pass
            
        spec = self.registry.get("f")
        self.assertEqual(spec.description, "Get weather.")

    def test_decorator_transparency(self):
        @tiny_tool(registry=self.registry)
        def add(a: int, b: int) -> int:
            return a + b
            
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add.__name__, "add")
