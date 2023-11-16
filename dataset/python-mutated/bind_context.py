from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.reader import ReaderLike2, ReaderLike3
from returns.primitives.hkt import Kind2, Kind3, Kinded, kinded
if TYPE_CHECKING:
    from returns.context import RequiresContext
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_Reader2Kind = TypeVar('_Reader2Kind', bound=ReaderLike2)
_Reader3Kind = TypeVar('_Reader3Kind', bound=ReaderLike3)

def bind_context2(function: Callable[[_FirstType], RequiresContext[_UpdatedType, _SecondType]]) -> Kinded[Callable[[Kind2[_Reader2Kind, _FirstType, _SecondType]], Kind2[_Reader2Kind, _UpdatedType, _SecondType]]]:
    if False:
        while True:
            i = 10
    "\n    Composes successful container with a function that returns a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> RequresContext[b, c]``\n    to:\n    ``Container[a, c] -> Container[b, c]``\n\n    .. code:: python\n\n      >>> from returns.pointfree import bind_context2\n      >>> from returns.context import Reader\n\n      >>> def example(argument: int) -> Reader[int, int]:\n      ...     return Reader(lambda deps: argument + deps)\n\n      >>> assert bind_context2(example)(Reader.from_value(2))(3) == 5\n\n    Note, that this function works with only ``Kind2`` containers\n    with ``.bind_context`` method.\n    See :class:`returns.primitives.interfaces.specific.reader.ReaderLike2`\n    for more info.\n\n    "

    @kinded
    def factory(container: Kind2[_Reader2Kind, _FirstType, _SecondType]) -> Kind2[_Reader2Kind, _UpdatedType, _SecondType]:
        if False:
            i = 10
            return i + 15
        return container.bind_context(function)
    return factory

def bind_context3(function: Callable[[_FirstType], RequiresContext[_UpdatedType, _ThirdType]]) -> Kinded[Callable[[Kind3[_Reader3Kind, _FirstType, _SecondType, _ThirdType]], Kind3[_Reader3Kind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        i = 10
        return i + 15
    "\n    Composes successful container with a function that returns a container.\n\n    In other words, it modifies the function's\n    signature from: ``a -> RequresContext[b, c]``\n    to: ``Container[a, c] -> Container[b, c]``\n\n    .. code:: python\n\n        >>> from returns.context import RequiresContext, RequiresContextResult\n        >>> from returns.result import Success, Failure\n        >>> from returns.pointfree import bind_context\n\n        >>> def function(arg: int) -> RequiresContext[str, int]:\n        ...     return RequiresContext(lambda deps: len(deps) + arg)\n\n        >>> assert bind_context(function)(\n        ...     RequiresContextResult.from_value(2),\n        ... )('abc') == Success(5)\n        >>> assert bind_context(function)(\n        ...     RequiresContextResult.from_failure(0),\n        ... )('abc') == Failure(0)\n\n    Note, that this function works with only ``Kind3`` containers\n    with ``.bind_context`` method.\n    See :class:`returns.primitives.interfaces.specific.reader.ReaderLike3`\n    for more info.\n\n    "

    @kinded
    def factory(container: Kind3[_Reader3Kind, _FirstType, _SecondType, _ThirdType]) -> Kind3[_Reader3Kind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            while True:
                i = 10
        return container.bind_context(function)
    return factory
bind_context = bind_context3