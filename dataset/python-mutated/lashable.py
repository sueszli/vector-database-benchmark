from abc import abstractmethod
from typing import Callable, Generic, NoReturn, TypeVar
from returns.primitives.hkt import KindN
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_LashableType = TypeVar('_LashableType', bound='LashableN')

class LashableN(Generic[_FirstType, _SecondType, _ThirdType]):
    """
    Represents a "context" in which calculations can be executed.

    ``Rescueable`` allows you to bind together
    a series of calculations while maintaining
    the context of that specific container.

    In contrast to :class:`returns.interfaces.bindable.BinbdaleN`,
    works with the second type value.
    """
    __slots__ = ()

    @abstractmethod
    def lash(self: _LashableType, function: Callable[[_SecondType], KindN[_LashableType, _FirstType, _UpdatedType, _ThirdType]]) -> KindN[_LashableType, _FirstType, _UpdatedType, _ThirdType]:
        if False:
            return 10
        "\n        Applies 'function' to the result of a previous calculation.\n\n        And returns a new container.\n        "
Lashable2 = LashableN[_FirstType, _SecondType, NoReturn]
Lashable3 = LashableN[_FirstType, _SecondType, _ThirdType]