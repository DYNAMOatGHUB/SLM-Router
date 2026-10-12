from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, Union

ParamType = Literal["str", "int", "float", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: ParamType
    required: bool
    default: Any = None
    optional: bool = False          # True when annotated Optional[T]; "None"/"null" coerces to None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    func: Callable[..., Any]


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    backend: str


class Backend(Protocol):
    name: str
    def complete(self, prompt: str, *, stop: Sequence[str], max_tokens: int) -> Completion: ...


@dataclass(frozen=True)
class ParsedAction:
    tool: str
    args: Mapping[str, str]         # raw, unquoted, unescaped; positional keys are "#0", "#1", ...
    raw: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedFinal:
    text: str
    raw: str


@dataclass(frozen=True)
class ParseFailure:
    code: str                       # one of errors.PARSE_CODES
    raw: str


ParseOutcome = Union[ParsedAction, ParsedFinal, ParseFailure]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    args: Mapping[str, Any]         # coerced values actually passed (empty if never called)
    output: str | None
    error_code: str | None
    error_detail: str | None        # short, single line, safe to show to the model
    duration_ms: float


@dataclass(frozen=True)
class AgentResult:
    status: Literal["tool", "final", "error"]
    text: str
    provenance: Literal["tool", "model", "none"]
    tool_result: ToolResult | None
    model_calls: int
    raw_outputs: tuple[str, ...]
    error_code: str | None
