from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.reader_ioresult import ReaderIOResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
if TYPE_CHECKING:
    from returns.context import ReaderIOResult
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ReaderIOResultLikeKind = TypeVar('_ReaderIOResultLikeKind', bound=ReaderIOResultLikeN)

def bind_context_ioresult(function: Callable[[_FirstType], ReaderIOResult[_UpdatedType, _SecondType, _ThirdType]]) -> Kinded[Callable[[KindN[_ReaderIOResultLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_ReaderIOResultLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        print('Hello World!')
    "\n    Lifts function from ``RequiresContextIOResult`` for better composition.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> RequiresContextIOResult[env, b, c]``\n    to:\n    ``Container[env, a, c]`` -> ``Container[env, b, c]``\n\n    .. code:: python\n\n      >>> import anyio\n      >>> from returns.context import (\n      ...     RequiresContextFutureResult,\n      ...     RequiresContextIOResult,\n      ... )\n      >>> from returns.io import IOSuccess, IOFailure\n      >>> from returns.pointfree import bind_context_ioresult\n\n      >>> def function(arg: int) -> RequiresContextIOResult[str, int, str]:\n      ...     return RequiresContextIOResult(\n      ...         lambda deps: IOSuccess(len(deps) + arg),\n      ...     )\n\n      >>> assert anyio.run(bind_context_ioresult(function)(\n      ...     RequiresContextFutureResult.from_value(2),\n      ... )('abc').awaitable) == IOSuccess(5)\n      >>> assert anyio.run(bind_context_ioresult(function)(\n      ...     RequiresContextFutureResult.from_failure(0),\n      ... )('abc').awaitable) == IOFailure(0)\n\n    "

    @kinded
    def factory(container: KindN[_ReaderIOResultLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_ReaderIOResultLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            while True:
                i = 10
        return container.bind_context_ioresult(function)
    return factory