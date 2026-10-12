"""Supported native local inference backends."""

from tinyagent.backends.llamacpp import LlamaCppBackend
from tinyagent.backends.ollama import OllamaBackend

__all__ = ["OllamaBackend", "LlamaCppBackend"]
