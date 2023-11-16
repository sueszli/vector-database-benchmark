from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.io import IOLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
if TYPE_CHECKING:
    from returns.io import IO
_FirstType = TypeVar('_FirstType', contravariant=True)
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType', contravariant=True)
_UpdatedType = TypeVar('_UpdatedType', covariant=True)
_IOLikeKind = TypeVar('_IOLikeKind', bound=IOLikeN)

def bind_io(function: Callable[[_FirstType], IO[_UpdatedType]]) -> Kinded[Callable[[KindN[_IOLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_IOLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        while True:
            i = 10
    "\n    Composes successful container with a function that returns a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> IO[b]``\n    to:\n    ``Container[a, c] -> Container[b, c]``\n\n    .. code:: python\n\n      >>> from returns.io import IOSuccess, IOFailure\n      >>> from returns.io import IO\n      >>> from returns.pointfree import bind_io\n\n      >>> def returns_io(arg: int) -> IO[int]:\n      ...     return IO(arg + 1)\n\n      >>> bound = bind_io(returns_io)\n      >>> assert bound(IO(1)) == IO(2)\n      >>> assert bound(IOSuccess(1)) == IOSuccess(2)\n      >>> assert bound(IOFailure(1)) == IOFailure(1)\n\n    "

    @kinded
    def factory(container: KindN[_IOLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_IOLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            while True:
                i = 10
        return container.bind_io(function)
    return factory