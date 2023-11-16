from typing import Awaitable, Callable, TypeVar
from returns.interfaces.specific.future import FutureLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_FutureKind = TypeVar('_FutureKind', bound=FutureLikeN)

def bind_awaitable(function: Callable[[_FirstType], Awaitable[_UpdatedType]]) -> Kinded[Callable[[KindN[_FutureKind, _FirstType, _SecondType, _ThirdType]], KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        while True:
            i = 10
    "\n    Composes a container a regular ``async`` function.\n\n    This function should return plain, non-container value.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> Awaitable[b]``\n    to:\n    ``Container[a] -> Container[b]``\n\n    This is how it should be used:\n\n    .. code:: python\n\n        >>> import anyio\n        >>> from returns.future import Future\n        >>> from returns.io import IO\n        >>> from returns.pointfree import bind_awaitable\n\n        >>> async def coroutine(x: int) -> int:\n        ...    return x + 1\n\n        >>> assert anyio.run(\n        ...     bind_awaitable(coroutine)(Future.from_value(1)).awaitable,\n        ... ) == IO(2)\n\n    Note, that this function works\n    for all containers with ``.bind_awaitable`` method.\n    See :class:`returns.primitives.interfaces.specific.future.FutureLikeN`\n    for more info.\n\n    "

    @kinded
    def factory(container: KindN[_FutureKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            i = 10
            return i + 15
        return container.bind_awaitable(function)
    return factory