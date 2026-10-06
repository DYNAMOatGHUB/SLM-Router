"""Ten-line offline quickstart."""

from examples.demo_tools import build_registry
from tinyagent import TinyAgent
from tinyagent.backends.simulator import SimulatedBackend

backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
agent = TinyAgent(backend, build_registry())
result = agent.run("What is 2+3?")
print(f"{result.text} (provenance: {result.provenance})")
