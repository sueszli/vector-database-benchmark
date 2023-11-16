from typing import Callable, TypeVar
from returns.interfaces.bimappable import BiMappableN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType1 = TypeVar('_UpdatedType1')
_UpdatedType2 = TypeVar('_UpdatedType2')
_BiMappableKind = TypeVar('_BiMappableKind', bound=BiMappableN)

def bimap(on_first: Callable[[_FirstType], _UpdatedType1], on_second: Callable[[_SecondType], _UpdatedType2]) -> Kinded[Callable[[KindN[_BiMappableKind, _FirstType, _SecondType, _ThirdType]], KindN[_BiMappableKind, _UpdatedType1, _UpdatedType2, _ThirdType]]]:
    if False:
        i = 10
        return i + 15
    "\n    Maps container on both: first and second arguments.\n\n    Can be used to synchronize state on both success and failure.\n\n    This is how it should be used:\n\n    .. code:: python\n\n        >>> from returns.io import IOSuccess, IOFailure\n        >>> from returns.pointfree import bimap\n\n        >>> def first(argument: int) -> float:\n        ...     return argument / 2\n\n        >>> def second(argument: str) -> bool:\n        ...     return bool(argument)\n\n        >>> assert bimap(first, second)(IOSuccess(1)) == IOSuccess(0.5)\n        >>> assert bimap(first, second)(IOFailure('')) == IOFailure(False)\n\n    Note, that this function works\n    for all containers with ``.map`` and ``.alt`` methods.\n    See :class:`returns.primitives.interfaces.bimappable.BiMappableN`\n    for more info.\n\n    "

    @kinded
    def factory(container: KindN[_BiMappableKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_BiMappableKind, _UpdatedType1, _UpdatedType2, _ThirdType]:
        if False:
            while True:
                i = 10
        return container.map(on_first).alt(on_second)
    return factory