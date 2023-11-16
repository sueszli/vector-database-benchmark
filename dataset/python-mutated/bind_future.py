from typing import Callable, TypeVar
from returns.future import Future
from returns.interfaces.specific.future import FutureLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_FutureKind = TypeVar('_FutureKind', bound=FutureLikeN)

def bind_future(function: Callable[[_FirstType], Future[_UpdatedType]]) -> Kinded[Callable[[KindN[_FutureKind, _FirstType, _SecondType, _ThirdType]], KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        i = 10
        return i + 15
    '\n    Compose a container and sync function returning ``Future``.\n\n    In other words, it modifies the function\n    signature from:\n    ``a -> Future[b]``\n    to:\n    ``Container[a] -> Container[b]``\n\n    Similar to :func:`returns.pointfree.lash`,\n    but works for successful containers.\n    This is how it should be used:\n\n    .. code:: python\n\n      >>> import anyio\n      >>> from returns.pointfree import bind_future\n      >>> from returns.future import Future\n      >>> from returns.io import IO\n\n      >>> def example(argument: int) -> Future[int]:\n      ...     return Future.from_value(argument + 1)\n\n      >>> assert anyio.run(\n      ...     bind_future(example)(Future.from_value(1)).awaitable,\n      ... ) == IO(2)\n\n    Note, that this function works\n    for all containers with ``.bind_future`` method.\n    See :class:`returns.primitives.interfaces.specific.future.FutureLikeN`\n    for more info.\n\n    '

    @kinded
    def factory(container: KindN[_FutureKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_FutureKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            i = 10
            return i + 15
        return container.bind_future(function)
    return factory