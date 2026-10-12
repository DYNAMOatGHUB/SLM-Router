import inspect
import re
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from .types import ParamSpec, ToolSpec
from .registry import ToolRegistry, default_registry

_ALLOWED_TYPES = {str: "str", int: "int", float: "float", bool: "bool"}

_UNSUPPORTED_KINDS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
    inspect.Parameter.POSITIONAL_ONLY,
)
_NAME_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _parse_type_annotation(param_name: str, fn_name: str, annotation: Any) -> tuple[str, bool]:
    if annotation is inspect.Parameter.empty:
        return "str", False

    # Handle Union[T, None] and T | None
    if get_origin(annotation) is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args and non_none[0] in _ALLOWED_TYPES:
            return _ALLOWED_TYPES[non_none[0]], True
    elif annotation in _ALLOWED_TYPES:
        return _ALLOWED_TYPES[annotation], False

    raise TypeError(
        f"tiny_tool: parameter '{param_name}' of '{fn_name}' has unsupported type "
        f"{annotation}; allowed: str,int,float,bool"
    )


def _describe(func: Callable, fallback: str) -> str:
    """First non-blank docstring line, whitespace-collapsed and capped at 80 chars."""
    doc = inspect.getdoc(func) or ""
    first = next((line for line in doc.splitlines() if line.strip()), None)
    if first is None:
        return fallback
    desc = " ".join(first.split())
    return desc[:77] + "..." if len(desc) > 80 else desc


def tiny_tool(func: Callable | None = None, *, name: str | None = None, registry: ToolRegistry | None = None):
    if func is None:
        def wrapper(f: Callable) -> Callable:
            return tiny_tool(f, name=name, registry=registry)
        return wrapper

    fn_name = func.__name__
    tool_name = name if name is not None else fn_name

    if not _NAME_REGEX.match(tool_name):
        raise ValueError(f"Invalid tool name: '{tool_name}'")

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    params = []
    for param_name, param in sig.parameters.items():
        if param.kind in _UNSUPPORTED_KINDS:
            raise TypeError(
                f"tiny_tool: parameter '{param_name}' of '{fn_name}' has unsupported kind {param.kind}"
            )

        annotation = type_hints.get(param_name, inspect.Parameter.empty)
        param_type, is_optional = _parse_type_annotation(param_name, fn_name, annotation)

        required = param.default is inspect.Parameter.empty

        params.append(ParamSpec(
            name=param_name,
            type=param_type,
            required=required,
            default=None if required else param.default,
            optional=is_optional,
        ))

    spec = ToolSpec(
        name=tool_name,
        description=_describe(func, tool_name),
        params=tuple(params),
        func=func,
    )
    (registry if registry is not None else default_registry).register(spec)
    return func
