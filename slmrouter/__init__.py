"""SLMRouter public API."""

__version__ = "0.1.0"
__all__ = ["tool", "SLMRouter", "ToolRegistry", "AgentResult"]


def __getattr__(name: str):
    if name == "tool":
        from slmrouter.reflection import tool

        return tool
    if name == "SLMRouter":
        from slmrouter.agent import SLMRouter

        return SLMRouter
    if name == "ToolRegistry":
        from slmrouter.registry import ToolRegistry

        return ToolRegistry
    if name == "AgentResult":
        from slmrouter.types import AgentResult

        return AgentResult
    raise AttributeError(name)
