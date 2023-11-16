from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.reader_result import ReaderResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
if TYPE_CHECKING:
    from returns.context import ReaderResult
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ReaderResultLikeKind = TypeVar('_ReaderResultLikeKind', bound=ReaderResultLikeN)

def bind_context_result(function: Callable[[_FirstType], ReaderResult[_UpdatedType, _SecondType, _ThirdType]]) -> Kinded[Callable[[KindN[_ReaderResultLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_ReaderResultLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Composes successful container with a function that returns a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> ReaderResult[b, c, e]``\n    to:\n    ``Container[a, c, e] -> Container[b, c, e]``\n\n    .. code:: python\n\n      >>> from returns.pointfree import bind_context_result\n      >>> from returns.context import ReaderIOResult, ReaderResult\n      >>> from returns.io import IOSuccess, IOFailure\n\n      >>> def example(argument: int) -> ReaderResult[int, str, str]:\n      ...     return ReaderResult.from_value(argument + 1)\n\n      >>> assert bind_context_result(example)(\n      ...     ReaderIOResult.from_value(1),\n      ... )(...) == IOSuccess(2)\n      >>> assert bind_context_result(example)(\n      ...     ReaderIOResult.from_failure('a'),\n      ... )(...) == IOFailure('a')\n\n    "

    @kinded
    def factory(container: KindN[_ReaderResultLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_ReaderResultLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            return 10
        return container.bind_context_result(function)
    return factory