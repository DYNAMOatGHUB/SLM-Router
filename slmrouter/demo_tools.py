"""Safe, deterministic demo tools used by the CLI, examples and evaluation harness.

These match the PRD-04 fixture signatures. They are demo fixtures, not part of
the stable public API.
"""

from __future__ import annotations


def build_registry():
    from slmrouter import ToolRegistry, tool

    registry = ToolRegistry()

    @tool(registry=registry)
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @tool(registry=registry)
    def get_weather(city: str, unit: str = "c") -> str:
        """Get fixture weather for a city."""
        return f"{city}: 31 {unit}, clear"

    @tool(registry=registry)
    def search(q: str, limit: int = 5) -> str:
        """Search fixture entries."""
        return f"{q}: {limit} results"

    @tool(registry=registry)
    def set_alarm(hour: int, minute: int, label: str = "") -> str:
        """Set a fixture alarm."""
        return f"Alarm {hour:02d}:{minute:02d} {label}".rstrip()

    @tool(registry=registry)
    def boom() -> None:
        """Raise a fixture error."""
        raise RuntimeError("fixture boom")

    return registry
