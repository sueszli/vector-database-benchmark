from typing import Callable, TypeVar
from returns.interfaces.applicative import ApplicativeN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ApplicativeKind = TypeVar('_ApplicativeKind', bound=ApplicativeN)

def apply(container: KindN[_ApplicativeKind, Callable[[_FirstType], _UpdatedType], _SecondType, _ThirdType]) -> Kinded[Callable[[KindN[_ApplicativeKind, _FirstType, _SecondType, _ThirdType]], KindN[_ApplicativeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Turns container containing a function into a callable.\n\n    In other words, it modifies the function\n    signature from:\n    ``Container[a -> b]``\n    to:\n    ``Container[a] -> Container[b]``\n\n    This is how it should be used:\n\n    .. code:: python\n\n      >>> from returns.pointfree import apply\n      >>> from returns.maybe import Some, Nothing\n\n      >>> def example(argument: int) -> int:\n      ...     return argument + 1\n\n      >>> assert apply(Some(example))(Some(1)) == Some(2)\n      >>> assert apply(Some(example))(Nothing) == Nothing\n      >>> assert apply(Nothing)(Some(1)) == Nothing\n      >>> assert apply(Nothing)(Nothing) == Nothing\n\n    Note, that this function works for all containers with ``.apply`` method.\n    See :class:`returns.interfaces.applicative.ApplicativeN` for more info.\n\n    '

    @kinded
    def factory(other: KindN[_ApplicativeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_ApplicativeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            while True:
                i = 10
        return other.apply(container)
    return factory