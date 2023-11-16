from typing import Callable, Tuple, TypeVar, final
from returns.interfaces import bindable, equable, lashable, swappable
from returns.primitives.container import BaseContainer, container_equality
from returns.primitives.hkt import Kind2, SupportsKind2, dekind
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_NewFirstType = TypeVar('_NewFirstType')
_NewSecondType = TypeVar('_NewSecondType')

@final
class Pair(BaseContainer, SupportsKind2['Pair', _FirstType, _SecondType], bindable.Bindable2[_FirstType, _SecondType], swappable.Swappable2[_FirstType, _SecondType], lashable.Lashable2[_FirstType, _SecondType], equable.Equable):
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
            print('Hello World!')
        return Pair((function(self._inner_value[0]), self._inner_value[1]))

    def bind(self, function: Callable[[_FirstType], Kind2['Pair', _NewFirstType, _SecondType]]) -> 'Pair[_NewFirstType, _SecondType]':
        if False:
            while True:
                i = 10
        return dekind(function(self._inner_value[0]))

    def alt(self, function: Callable[[_SecondType], _NewSecondType]) -> 'Pair[_FirstType, _NewSecondType]':
        if False:
            print('Hello World!')
        return Pair((self._inner_value[0], function(self._inner_value[1])))

    def lash(self, function: Callable[[_SecondType], Kind2['Pair', _FirstType, _NewSecondType]]) -> 'Pair[_FirstType, _NewSecondType]':
        if False:
            while True:
                i = 10
        return dekind(function(self._inner_value[1]))

    def swap(self) -> 'Pair[_SecondType, _FirstType]':
        if False:
            i = 10
            return i + 15
        return Pair((self._inner_value[1], self._inner_value[0]))