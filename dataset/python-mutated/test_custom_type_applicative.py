from typing import Callable, TypeVar
from returns.contrib.hypothesis.laws import check_all_laws
from returns.interfaces import applicative
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1
_ValueType = TypeVar('_ValueType')
_NewValueType = TypeVar('_NewValueType')

class _Wrapper(BaseContainer, SupportsKind1['_Wrapper', _ValueType], applicative.Applicative1[_ValueType]):
    _inner_value: _ValueType

    def __init__(self, inner_value: _ValueType) -> None:
        if False:
            print('Hello World!')
        super().__init__(inner_value)

    def map(self, function: Callable[[_ValueType], _NewValueType]) -> '_Wrapper[_NewValueType]':
        if False:
            i = 10
            return i + 15
        return _Wrapper(function(self._inner_value))

    def apply(self, container: Kind1['_Wrapper', Callable[[_ValueType], _NewValueType]]) -> '_Wrapper[_NewValueType]':
        if False:
            while True:
                i = 10
        function = container._inner_value
        return _Wrapper(function(self._inner_value))

    @classmethod
    def from_value(cls, inner_value: _NewValueType) -> '_Wrapper[_NewValueType]':
        if False:
            for i in range(10):
                print('nop')
        return _Wrapper(inner_value)
check_all_laws(_Wrapper)