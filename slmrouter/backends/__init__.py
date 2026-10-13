"""Supported native local inference backends."""

from slmrouter.backends.llamacpp import LlamaCppBackend
from slmrouter.backends.ollama import OllamaBackend

__all__ = ["OllamaBackend", "LlamaCppBackend"]
