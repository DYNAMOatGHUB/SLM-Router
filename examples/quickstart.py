"""Ten-line quickstart: python examples/quickstart.py [--backend sim|ollama]."""

import sys

from slmrouter import SLMRouter
from slmrouter.backends import OllamaBackend
from slmrouter.backends.simulator import SimulatedBackend
from slmrouter.demo_tools import build_registry

if "ollama" in sys.argv:
    backend = OllamaBackend("llama3:8b")
else:
    backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
result = SLMRouter(backend, build_registry()).run("What is 2+3?")
print(f"{result.text} (provenance: {result.provenance})")
