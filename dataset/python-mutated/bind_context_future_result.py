from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.reader_future_result import ReaderFutureResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
if TYPE_CHECKING:
    from returns.context import ReaderFutureResult
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ReaderFutureResultLikeKind = TypeVar('_ReaderFutureResultLikeKind', bound=ReaderFutureResultLikeN)

def bind_context_future_result(function: Callable[[_FirstType], 'ReaderFutureResult[_UpdatedType, _SecondType, _ThirdType]']) -> Kinded[Callable[[KindN[_ReaderFutureResultLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_ReaderFutureResultLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        while True:
            i = 10
    "\n    Lifts function from ``RequiresContextFutureResult`` for better composition.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> RequiresContextFutureResult[env, b, c]``\n    to:\n    ``Container[env, a, c]`` -> ``Container[env, b, c]``\n\n    .. code:: python\n\n      >>> import anyio\n      >>> from returns.context import ReaderFutureResult\n      >>> from returns.io import IOSuccess, IOFailure\n      >>> from returns.future import FutureResult\n      >>> from returns.pointfree import bind_context_future_result\n\n      >>> def function(arg: int) -> ReaderFutureResult[str, int, str]:\n      ...     return ReaderFutureResult(\n      ...         lambda deps: FutureResult.from_value(len(deps) + arg),\n      ...     )\n\n      >>> assert anyio.run(bind_context_future_result(function)(\n      ...     ReaderFutureResult.from_value(2),\n      ... )('abc').awaitable) == IOSuccess(5)\n      >>> assert anyio.run(bind_context_future_result(function)(\n      ...     ReaderFutureResult.from_failure(0),\n      ... )('abc').awaitable) == IOFailure(0)\n\n    "

    @kinded
    def factory(container: KindN[_ReaderFutureResultLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_ReaderFutureResultLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            return 10
        return container.bind_context_future_result(function)
    return factory