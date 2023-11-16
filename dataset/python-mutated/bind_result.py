from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar
from returns.interfaces.specific.result import ResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
if TYPE_CHECKING:
    from returns.result import Result
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ResultLikeKind = TypeVar('_ResultLikeKind', bound=ResultLikeN)

def bind_result(function: Callable[[_FirstType], Result[_UpdatedType, _SecondType]]) -> Kinded[Callable[[KindN[_ResultLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_ResultLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        while True:
            i = 10
    "\n    Composes successful container with a function that returns a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> Result[b, c]``\n    to:\n    ``Container[a, c] -> Container[b, c]``\n\n    .. code:: python\n\n      >>> from returns.io import IOSuccess\n      >>> from returns.context import RequiresContextResult\n      >>> from returns.result import Result, Success\n      >>> from returns.pointfree import bind_result\n\n      >>> def returns_result(arg: int) -> Result[int, str]:\n      ...     return Success(arg + 1)\n\n      >>> bound = bind_result(returns_result)\n      >>> assert bound(IOSuccess(1)) == IOSuccess(2)\n      >>> assert bound(RequiresContextResult.from_value(1))(...) == Success(2)\n\n    "

    @kinded
    def factory(container: KindN[_ResultLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_ResultLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            while True:
                i = 10
        return container.bind_result(function)
    return factory