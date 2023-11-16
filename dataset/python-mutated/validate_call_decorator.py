"""Decorator for validating function calls."""
from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from ._internal import _validate_call
__all__ = ('validate_call',)
if TYPE_CHECKING:
    from .config import ConfigDict
    AnyCallableT = TypeVar('AnyCallableT', bound=Callable[..., Any])

@overload
def validate_call(*, config: ConfigDict | None=None, validate_return: bool=False) -> Callable[[AnyCallableT], AnyCallableT]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def validate_call(__func: AnyCallableT) -> AnyCallableT:
    if False:
        i = 10
        return i + 15
    ...

def validate_call(__func: AnyCallableT | None=None, *, config: ConfigDict | None=None, validate_return: bool=False) -> AnyCallableT | Callable[[AnyCallableT], AnyCallableT]:
    if False:
        i = 10
        return i + 15
    'Usage docs: https://docs.pydantic.dev/2.6/concepts/validation_decorator/\n\n    Returns a decorated wrapper around the function that validates the arguments and, optionally, the return value.\n\n    Usage may be either as a plain decorator `@validate_call` or with arguments `@validate_call(...)`.\n\n    Args:\n        __func: The function to be decorated.\n        config: The configuration dictionary.\n        validate_return: Whether to validate the return value.\n\n    Returns:\n        The decorated function.\n    '

    def validate(function: AnyCallableT) -> AnyCallableT:
        if False:
            i = 10
            return i + 15
        if isinstance(function, (classmethod, staticmethod)):
            name = type(function).__name__
            raise TypeError(f'The `@{name}` decorator should be applied after `@validate_call` (put `@{name}` on top)')
        return _validate_call.ValidateCallWrapper(function, config, validate_return)
    if __func:
        return validate(__func)
    else:
        return validate