import unittest
from tinyagent.registry import ToolRegistry
from tinyagent.types import ToolSpec

class TestRegistry(unittest.TestCase):
    def test_registry_basics(self):
        reg = ToolRegistry()
        self.assertEqual(len(reg), 0)
        
        spec1 = ToolSpec(name="tool1", description="desc1", params=(), func=lambda: None)
        reg.register(spec1)
        
        self.assertEqual(len(reg), 1)
        self.assertIn("tool1", reg)
        self.assertNotIn("tool2", reg)
        
        self.assertEqual(reg.get("tool1"), spec1)
        self.assertIsNone(reg.get("tool2"))
        
        spec2 = ToolSpec(name="tool2", description="desc2", params=(), func=lambda: None)
        reg.register(spec2)
        
        self.assertEqual(reg.all(), (spec1, spec2))
        self.assertEqual(reg.names(), ("tool1", "tool2"))
        
    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        spec1 = ToolSpec(name="tool1", description="desc1", params=(), func=lambda: None)
        reg.register(spec1)
        
        spec2 = ToolSpec(name="tool1", description="desc2", params=(), func=lambda: None)
        with self.assertRaises(ValueError):
            reg.register(spec2)
