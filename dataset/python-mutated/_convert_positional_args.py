from __future__ import annotations
from collections.abc import Callable
from collections.abc import Sequence
from functools import wraps
from inspect import Parameter
from inspect import signature
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings
if TYPE_CHECKING:
    from typing_extensions import ParamSpec
    _P = ParamSpec('_P')
    _T = TypeVar('_T')

def _get_positional_arg_names(func: 'Callable[_P, _T]') -> list[str]:
    if False:
        print('Hello World!')
    params = signature(func).parameters
    positional_arg_names = [name for (name, p) in params.items() if p.default == Parameter.empty and p.kind == p.POSITIONAL_OR_KEYWORD]
    return positional_arg_names

def _infer_kwargs(previous_positional_arg_names: Sequence[str], *args: Any) -> dict[str, Any]:
    if False:
        return 10
    inferred_kwargs = {arg_name: val for (val, arg_name) in zip(args, previous_positional_arg_names)}
    return inferred_kwargs

def convert_positional_args(*, previous_positional_arg_names: Sequence[str], warning_stacklevel: int=2) -> 'Callable[[Callable[_P, _T]], Callable[_P, _T]]':
    if False:
        i = 10
        return i + 15
    'Convert positional arguments to keyword arguments.\n\n    Args:\n        previous_positional_arg_names: List of names previously given as positional arguments.\n        warning_stacklevel: Level of the stack trace where decorated function locates.\n    '

    def converter_decorator(func: 'Callable[_P, _T]') -> 'Callable[_P, _T]':
        if False:
            return 10
        assert set(previous_positional_arg_names).issubset(set(signature(func).parameters)), f'{set(previous_positional_arg_names)} is not a subset of {set(signature(func).parameters)}'

        @wraps(func)
        def converter_wrapper(*args: Any, **kwargs: Any) -> '_T':
            if False:
                for i in range(10):
                    print('nop')
            positional_arg_names = _get_positional_arg_names(func)
            inferred_kwargs = _infer_kwargs(previous_positional_arg_names, *args)
            if len(inferred_kwargs) > len(positional_arg_names):
                expected_kwds = set(inferred_kwargs) - set(positional_arg_names)
                warnings.warn(f'{func.__name__}() got {expected_kwds} as positional arguments but they were expected to be given as keyword arguments.', FutureWarning, stacklevel=warning_stacklevel)
            if len(args) > len(previous_positional_arg_names):
                raise TypeError(f'{func.__name__}() takes {len(previous_positional_arg_names)} positional arguments but {len(args)} were given.')
            duplicated_kwds = set(kwargs).intersection(inferred_kwargs)
            if len(duplicated_kwds):
                raise TypeError(f'{func.__name__}() got multiple values for arguments {duplicated_kwds}.')
            kwargs.update(inferred_kwargs)
            return func(**kwargs)
        return converter_wrapper
    return converter_decorator