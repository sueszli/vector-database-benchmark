from typing import Awaitable, Callable, TypeVar
from returns.future import FutureResult
from returns.interfaces.specific.future_result import FutureResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_FutureResultKind = TypeVar('_FutureResultKind', bound=FutureResultLikeN)

def bind_async_future_result(function: Callable[[_FirstType], Awaitable[FutureResult[_UpdatedType, _SecondType]]]) -> Kinded[Callable[[KindN[_FutureResultKind, _FirstType, _SecondType, _ThirdType]], KindN[_FutureResultKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        print('Hello World!')
    "\n    Compose a container and async function returning ``FutureResult``.\n\n    In other words, it modifies the function\n    signature from:\n    ``a -> Awaitable[FutureResult[b, c]]``\n    to:\n    ``Container[a, c] -> Container[b, c]``\n\n    This is how it should be used:\n\n    .. code:: python\n\n      >>> import anyio\n      >>> from returns.pointfree import bind_async_future_result\n      >>> from returns.future import FutureResult\n      >>> from returns.io import IOSuccess, IOFailure\n\n      >>> async def example(argument: int) -> FutureResult[int, str]:\n      ...     return FutureResult.from_value(argument + 1)\n\n      >>> assert anyio.run(\n      ...     bind_async_future_result(example)(\n      ...         FutureResult.from_value(1),\n      ...     ).awaitable,\n      ... ) == IOSuccess(2)\n\n      >>> assert anyio.run(\n      ...     bind_async_future_result(example)(\n      ...         FutureResult.from_failure('a'),\n      ...     ).awaitable,\n      ... ) == IOFailure('a')\n\n    .. currentmodule: returns.primitives.interfaces.specific.future_result\n\n    Note, that this function works\n    for all containers with ``.bind_async_future`` method.\n    See :class:`~FutureResultLikeN` for more info.\n\n    "

    @kinded
    def factory(container: KindN[_FutureResultKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_FutureResultKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            for i in range(10):
                print('nop')
        return container.bind_async_future_result(function)
    return factory