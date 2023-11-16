from abc import abstractmethod
from typing import Callable, NoReturn, Tuple, Type, TypeVar, final
from returns.interfaces import bindable, equable, lashable, swappable
from returns.primitives.container import BaseContainer, container_equality
from returns.primitives.hkt import Kind2, KindN, SupportsKind2, dekind
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_NewFirstType = TypeVar('_NewFirstType')
_NewSecondType = TypeVar('_NewSecondType')
_PairLikeKind = TypeVar('_PairLikeKind', bound='PairLikeN')

class PairLikeN(bindable.BindableN[_FirstType, _SecondType, _ThirdType], swappable.SwappableN[_FirstType, _SecondType, _ThirdType], lashable.LashableN[_FirstType, _SecondType, _ThirdType], equable.Equable):
    """Special interface for types that look like a ``Pair``."""

    @abstractmethod
    def pair(self: _PairLikeKind, function: Callable[[_FirstType, _SecondType], KindN[_PairLikeKind, _NewFirstType, _NewSecondType, _ThirdType]]) -> KindN[_PairLikeKind, _NewFirstType, _NewSecondType, _ThirdType]:
        if False:
            print('Hello World!')
        'Allows to work with both arguments at the same time.'

    @classmethod
    @abstractmethod
    def from_paired(cls: Type[_PairLikeKind], first: _NewFirstType, second: _NewSecondType) -> KindN[_PairLikeKind, _NewFirstType, _NewSecondType, _ThirdType]:
        if False:
            print('Hello World!')
        'Allows to create a PairLikeN from just two values.'

    @classmethod
    @abstractmethod
    def from_unpaired(cls: Type[_PairLikeKind], inner_value: _NewFirstType) -> KindN[_PairLikeKind, _NewFirstType, _NewFirstType, _ThirdType]:
        if False:
            for i in range(10):
                print('nop')
        'Allows to create a PairLikeN from just a single object.'
PairLike2 = PairLikeN[_FirstType, _SecondType, NoReturn]
PairLike3 = PairLikeN[_FirstType, _SecondType, _ThirdType]

@final
class Pair(BaseContainer, SupportsKind2['Pair', _FirstType, _SecondType], PairLike2[_FirstType, _SecondType]):
    """
    A type that represents a pair of something.

    Like to coordinates ``(x, y)`` or two best friends.
    Or a question and an answer.

    """

    def __init__(self, inner_value: Tuple[_FirstType, _SecondType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Saves passed tuple as ``._inner_value`` inside this instance.'
        super().__init__(inner_value)
    equals = container_equality

    def map(self, function: Callable[[_FirstType], _NewFirstType]) -> 'Pair[_NewFirstType, _SecondType]':
        if False:
            i = 10
            return i + 15
        "\n        Changes the first type with a pure function.\n\n        >>> assert Pair((1, 2)).map(str) == Pair(('1', 2))\n\n        "
        return Pair((function(self._inner_value[0]), self._inner_value[1]))

    def bind(self, function: Callable[[_FirstType], Kind2['Pair', _NewFirstType, _SecondType]]) -> 'Pair[_NewFirstType, _SecondType]':
        if False:
            for i in range(10):
                print('nop')
        "\n        Changes the first type with a function returning another ``Pair``.\n\n        >>> def bindable(first: int) -> Pair[str, str]:\n        ...     return Pair((str(first), ''))\n\n        >>> assert Pair((1, 'b')).bind(bindable) == Pair(('1', ''))\n\n        "
        return dekind(function(self._inner_value[0]))

    def alt(self, function: Callable[[_SecondType], _NewSecondType]) -> 'Pair[_FirstType, _NewSecondType]':
        if False:
            for i in range(10):
                print('nop')
        "\n        Changes the second type with a pure function.\n\n        >>> assert Pair((1, 2)).alt(str) == Pair((1, '2'))\n\n        "
        return Pair((self._inner_value[0], function(self._inner_value[1])))

    def lash(self, function: Callable[[_SecondType], Kind2['Pair', _FirstType, _NewSecondType]]) -> 'Pair[_FirstType, _NewSecondType]':
        if False:
            for i in range(10):
                print('nop')
        "\n        Changes the second type with a function returning ``Pair``.\n\n        >>> def lashable(second: int) -> Pair[str, str]:\n        ...     return Pair(('', str(second)))\n\n        >>> assert Pair(('a', 2)).lash(lashable) == Pair(('', '2'))\n\n        "
        return dekind(function(self._inner_value[1]))

    def swap(self) -> 'Pair[_SecondType, _FirstType]':
        if False:
            print('Hello World!')
        '\n        Swaps ``Pair`` elements.\n\n        >>> assert Pair((1, 2)).swap() == Pair((2, 1))\n\n        '
        return Pair((self._inner_value[1], self._inner_value[0]))

    def pair(self, function: Callable[[_FirstType, _SecondType], Kind2['Pair', _NewFirstType, _NewSecondType]]) -> 'Pair[_NewFirstType, _NewSecondType]':
        if False:
            i = 10
            return i + 15
        '\n        Creates a new ``Pair`` from an existing one via a passed function.\n\n        >>> def min_max(first: int, second: int) -> Pair[int, int]:\n        ...     return Pair((min(first, second), max(first, second)))\n\n        >>> assert Pair((2, 1)).pair(min_max) == Pair((1, 2))\n        >>> assert Pair((1, 2)).pair(min_max) == Pair((1, 2))\n\n        '
        return dekind(function(self._inner_value[0], self._inner_value[1]))

    @classmethod
    def from_paired(cls, first: _NewFirstType, second: _NewSecondType) -> 'Pair[_NewFirstType, _NewSecondType]':
        if False:
            i = 10
            return i + 15
        '\n        Creates a new pair from two values.\n\n        >>> assert Pair.from_paired(1, 2) == Pair((1, 2))\n\n        '
        return Pair((first, second))

    @classmethod
    def from_unpaired(cls, inner_value: _NewFirstType) -> 'Pair[_NewFirstType, _NewFirstType]':
        if False:
            while True:
                i = 10
        '\n        Creates a new pair from a single value.\n\n        >>> assert Pair.from_unpaired(1) == Pair((1, 1))\n\n        '
        return Pair((inner_value, inner_value))