from typing import Callable, TypeVar
from returns.interfaces.altable import AltableN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_AltableKind = TypeVar('_AltableKind', bound=AltableN)

def alt(function: Callable[[_SecondType], _UpdatedType]) -> Kinded[Callable[[KindN[_AltableKind, _FirstType, _SecondType, _ThirdType]], KindN[_AltableKind, _FirstType, _UpdatedType, _ThirdType]]]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Lifts function to be wrapped in a container for better composition.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> b``\n    to:\n    ``Container[a] -> Container[b]``\n\n    This is how it should be used:\n\n    .. code:: python\n\n        >>> from returns.io import IOFailure, IOSuccess\n        >>> from returns.pointfree import alt\n\n        >>> def example(argument: int) -> float:\n        ...     return argument / 2\n\n        >>> assert alt(example)(IOSuccess(1)) == IOSuccess(1)\n        >>> assert alt(example)(IOFailure(4)) == IOFailure(2.0)\n\n    Note, that this function works for all containers with ``.alt`` method.\n    See :class:`returns.primitives.interfaces.altable.AltableN` for more info.\n\n    "

    @kinded
    def factory(container: KindN[_AltableKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_AltableKind, _FirstType, _UpdatedType, _ThirdType]:
        if False:
            while True:
                i = 10
        return container.alt(function)
    return factory