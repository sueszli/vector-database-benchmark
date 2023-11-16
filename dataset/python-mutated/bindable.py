from abc import abstractmethod
from typing import Callable, Generic, NoReturn, TypeVar
from returns.primitives.hkt import KindN
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_BindableType = TypeVar('_BindableType', bound='BindableN')

class BindableN(Generic[_FirstType, _SecondType, _ThirdType]):
    """
    Represents a "context" in which calculations can be executed.

    ``Bindable`` allows you to bind together
    a series of calculations while maintaining
    the context of that specific container.

    In contrast to :class:`returns.interfaces.lashable.LashableN`,
    works with the first type argument.
    """
    __slots__ = ()

    @abstractmethod
    def bind(self: _BindableType, function: Callable[[_FirstType], KindN[_BindableType, _UpdatedType, _SecondType, _ThirdType]]) -> KindN[_BindableType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            return 10
        "\n        Applies 'function' to the result of a previous calculation.\n\n        And returns a new container.\n        "
Bindable1 = BindableN[_FirstType, NoReturn, NoReturn]
Bindable2 = BindableN[_FirstType, _SecondType, NoReturn]
Bindable3 = BindableN[_FirstType, _SecondType, _ThirdType]