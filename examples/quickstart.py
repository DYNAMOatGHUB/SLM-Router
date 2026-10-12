"""Ten-line quickstart: python examples/quickstart.py [--backend sim|ollama]."""

import sys

from tinyagent import TinyAgent
from tinyagent.backends import OllamaBackend
from tinyagent.backends.simulator import SimulatedBackend
from tinyagent.demo_tools import build_registry

if "ollama" in sys.argv:
    backend = OllamaBackend("llama3:8b")
else:
    backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
result = TinyAgent(backend, build_registry()).run("What is 2+3?")
print(f"{result.text} (provenance: {result.provenance})")
