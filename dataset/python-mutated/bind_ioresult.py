from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.ioresult import IOResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
if TYPE_CHECKING:
    from returns.io import IOResult
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_IOResultLikeKind = TypeVar('_IOResultLikeKind', bound=IOResultLikeN)

def bind_ioresult(function: Callable[[_FirstType], IOResult[_UpdatedType, _SecondType]]) -> Kinded[Callable[[KindN[_IOResultLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_IOResultLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        return 10
    "\n    Composes successful container with a function that returns a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> IOResult[b, c]``\n    to:\n    ``Container[a, c] -> Container[b, c]``\n\n    .. code:: python\n\n      >>> from returns.io import IOResult, IOSuccess\n      >>> from returns.context import RequiresContextIOResult\n      >>> from returns.pointfree import bind_ioresult\n\n      >>> def returns_ioresult(arg: int) -> IOResult[int, str]:\n      ...     return IOSuccess(arg + 1)\n\n      >>> bound = bind_ioresult(returns_ioresult)\n      >>> assert bound(IOSuccess(1)) == IOSuccess(2)\n      >>> assert bound(\n      ...     RequiresContextIOResult.from_value(1),\n      ... )(...) == IOSuccess(2)\n\n    "

    @kinded
    def factory(container: KindN[_IOResultLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_IOResultLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            i = 10
            return i + 15
        return container.bind_ioresult(function)
    return factory