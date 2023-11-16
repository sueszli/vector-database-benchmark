from abc import abstractmethod
from typing import Generic, TypeVar
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_UnwrappableType = TypeVar('_UnwrappableType', bound='Unwrappable')

class Unwrappable(Generic[_FirstType, _SecondType]):
    """
    Represents containers that can unwrap and return its wrapped value.

    There are no aliases or ``UnwrappableN`` for ``Unwrappable`` interface.
    Because it always uses two and just two types.

    Not all types can be ``Unwrappable`` because we do require
    to raise ``UnwrapFailedError`` if unwrap is not possible.
    """
    __slots__ = ()

    @abstractmethod
    def unwrap(self: _UnwrappableType) -> _FirstType:
        if False:
            return 10
        '\n        Custom magic method to unwrap inner value from container.\n\n        Should be redefined for ones that actually have values.\n        And for ones that raise an exception for no values.\n\n        .. note::\n            As a part of the contract, failed ``unwrap`` calls\n            must raise :class:`returns.primitives.exceptions.UnwrapFailedError`\n            exception.\n\n        This method is the opposite of :meth:`~Unwrapable.failure`.\n        '

    @abstractmethod
    def failure(self: _UnwrappableType) -> _SecondType:
        if False:
            i = 10
            return i + 15
        '\n        Custom magic method to unwrap inner value from the failed container.\n\n        .. note::\n            As a part of the contract, failed ``failure`` calls\n            must raise :class:`returns.primitives.exceptions.UnwrapFailedError`\n            exception.\n\n        This method is the opposite of :meth:`~Unwrapable.unwrap`.\n        '