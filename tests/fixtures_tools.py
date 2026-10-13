from slmrouter.reflection import tool
from slmrouter.registry import ToolRegistry

fixture_registry = ToolRegistry()

@tool(registry=fixture_registry)
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

@tool(registry=fixture_registry)
def get_weather(city: str, unit: str = "c") -> str:
    """Current weather for a city."""
    return f"Weather in {city} in {unit}"

@tool(registry=fixture_registry)
def search(query: str, limit: int = 10) -> str:
    """Search the web for a query."""
    return f"Search results for {query}"

@tool(registry=fixture_registry)
def set_alarm(time: str, label: str = "Alarm", active: bool = True) -> bool:
    """Set an alarm clock."""
    return True
