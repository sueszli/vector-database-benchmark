from typing import Awaitable, Callable, TypeVar
from returns.interfaces.specific.future import FutureLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_FutureKind = TypeVar('_FutureKind', bound=FutureLikeN)

def bind_async(function: Callable[[_FirstType], Awaitable[KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]]]) -> Kinded[Callable[[KindN[_FutureKind, _FirstType, _SecondType, _ThirdType]], KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        while True:
            i = 10
    "\n    Compose a container and ``async`` function returning a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> Awaitable[Container[b]]``\n    to:\n    ``Container[a] -> Container[b]``\n\n    This is how it should be used:\n\n    .. code:: python\n\n        >>> import anyio\n        >>> from returns.future import Future\n        >>> from returns.io import IO\n        >>> from returns.pointfree import bind_async\n\n        >>> async def coroutine(x: int) -> Future[str]:\n        ...    return Future.from_value(str(x + 1))\n\n        >>> bound = bind_async(coroutine)(Future.from_value(1))\n        >>> assert anyio.run(bound.awaitable) == IO('2')\n\n    Note, that this function works\n    for all containers with ``.bind_async`` method.\n    See :class:`returns.primitives.interfaces.specific.future.FutureLikeN`\n    for more info.\n\n    "

    @kinded
    def factory(container: KindN[_FutureKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            return 10
        return container.bind_async(function)
    return factory