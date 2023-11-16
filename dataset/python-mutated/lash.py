from typing import Callable, TypeVar
from returns.interfaces.lashable import LashableN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_LashableKind = TypeVar('_LashableKind', bound=LashableN)

def lash(function: Callable[[_SecondType], KindN[_LashableKind, _FirstType, _UpdatedType, _ThirdType]]) -> Kinded[Callable[[KindN[_LashableKind, _FirstType, _SecondType, _ThirdType]], KindN[_LashableKind, _FirstType, _UpdatedType, _ThirdType]]]:
    if False:
        while True:
            i = 10
    "\n    Turns function's input parameter from a regular value to a container.\n\n    In other words, it modifies the function\n    signature from:\n    ``a -> Container[b]``\n    to:\n    ``Container[a] -> Container[b]``\n\n    Similar to :func:`returns.pointfree.bind`, but works for failed containers.\n\n    This is how it should be used:\n\n    .. code:: python\n\n      >>> from returns.pointfree import lash\n      >>> from returns.result import Success, Failure, Result\n\n      >>> def example(argument: int) -> Result[str, int]:\n      ...     return Success(argument + 1)\n\n      >>> assert lash(example)(Success('a')) == Success('a')\n      >>> assert lash(example)(Failure(1)) == Success(2)\n\n    Note, that this function works for all containers with ``.lash`` method.\n    See :class:`returns.interfaces.lashable.Lashable` for more info.\n\n    "

    @kinded
    def factory(container: KindN[_LashableKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_LashableKind, _FirstType, _UpdatedType, _ThirdType]:
        if False:
            return 10
        return container.lash(function)
    return factory