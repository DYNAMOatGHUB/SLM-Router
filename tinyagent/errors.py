PARSE_CODES = frozenset({"E_EMPTY", "E_NO_PREFIX", "E_BAD_SYNTAX"})
EXEC_REPAIRABLE = frozenset({"E_UNKNOWN_TOOL", "E_MISSING_ARG", "E_UNEXPECTED_ARG", "E_TYPE_COERCION"})
EXEC_TERMINAL = frozenset({"E_TOOL_EXCEPTION", "E_TOOL_TIMEOUT"})
BACKEND_CODES = frozenset({"E_BACKEND_UNREACHABLE", "E_BACKEND_TIMEOUT", "E_BACKEND_HTTP", "E_BACKEND_PROTOCOL"})
AGENT_CODES = frozenset({"E_FINAL_NOT_ALLOWED", "E_RETRY_EXHAUSTED"})

class TinyAgentError(Exception):
    """Base class. Only raised for programmer errors and backend failures, never for model output."""

class BackendError(TinyAgentError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

class CoercionError(TinyAgentError, ValueError):
    pass
