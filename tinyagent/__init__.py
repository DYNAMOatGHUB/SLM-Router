"""TinyAgent public API."""

__version__ = "0.1.0"
__all__ = ["tiny_tool", "TinyAgent", "ToolRegistry", "AgentResult"]


def __getattr__(name: str):
    if name == "tiny_tool":
        from tinyagent.reflection import tiny_tool

        return tiny_tool
    if name == "TinyAgent":
        from tinyagent.agent import TinyAgent

        return TinyAgent
    if name == "ToolRegistry":
        from tinyagent.registry import ToolRegistry

        return ToolRegistry
    if name == "AgentResult":
        from tinyagent.types import AgentResult

        return AgentResult
    raise AttributeError(name)
