import asyncio
import inspect
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, TypeVar
from typing_extensions import get_args
from strawberry.type import has_object_definition

def in_async_context() -> bool:
    if False:
        return 10
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return True

@lru_cache(maxsize=250)
def get_func_args(func: Callable[[Any], Any]) -> List[str]:
    if False:
        return 10
    'Returns a list of arguments for the function'
    sig = inspect.signature(func)
    return [arg_name for (arg_name, param) in sig.parameters.items() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]

def get_specialized_type_var_map(cls: type) -> Optional[Dict[str, type]]:
    if False:
        return 10
    'Get a type var map for specialized types.\n\n    Consider the following:\n\n        >>> class Foo(Generic[T]):\n        ...     ...\n        ...\n        >>> class Bar(Generic[K]):\n        ...     ...\n        ...\n        >>> class IntBar(Bar[int]):\n        ...     ...\n        ...\n        >>> class IntBarSubclass(IntBar):\n        ...     ...\n        ...\n        >>> class IntBarFoo(IntBar, Foo[str]):\n        ...     ...\n        ...\n\n    This would return:\n\n        >>> get_specialized_type_var_map(object)\n        None\n        >>> get_specialized_type_var_map(Foo)\n        {}\n        >>> get_specialized_type_var_map(Bar)\n        {~T: ~T}\n        >>> get_specialized_type_var_map(IntBar)\n        {~T: int}\n        >>> get_specialized_type_var_map(IntBarSubclass)\n        {~T: int}\n        >>> get_specialized_type_var_map(IntBarFoo)\n        {~T: int, ~K: str}\n\n    '
    orig_bases = getattr(cls, '__orig_bases__', None)
    if orig_bases is None:
        return None
    type_var_map = {}
    orig_bases = [b for b in orig_bases if has_object_definition(b)]
    for base in orig_bases:
        base_type_var_map = get_specialized_type_var_map(base)
        if base_type_var_map is not None:
            type_var_map.update(base_type_var_map)
        args = get_args(base)
        origin = getattr(base, '__origin__', None)
        params = origin and getattr(origin, '__parameters__', None)
        if params is None:
            params = getattr(base, '__parameters__', None)
        if not params:
            continue
        type_var_map.update({p.__name__: a for (p, a) in zip(params, args) if not isinstance(a, TypeVar)})
    return type_var_map