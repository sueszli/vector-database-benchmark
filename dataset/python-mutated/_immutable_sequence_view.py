"""Provides an immutable sequence view class."""
from __future__ import annotations
from sys import maxsize
from typing import Generic, Iterator, Sequence, TypeVar, overload
T = TypeVar('T')

class ImmutableSequenceView(Generic[T]):
    """Class to wrap a sequence of some sort, but not allow modification."""

    def __init__(self, wrap: Sequence[T]) -> None:
        if False:
            return 10
        'Initialise the immutable sequence.\n\n        Args:\n            wrap: The sequence being wrapped.\n        '
        self._wrap = wrap

    @overload
    def __getitem__(self, index: int) -> T:
        if False:
            return 10
        ...

    @overload
    def __getitem__(self, index: slice) -> ImmutableSequenceView[T]:
        if False:
            return 10
        ...

    def __getitem__(self, index: int | slice) -> T | ImmutableSequenceView[T]:
        if False:
            while True:
                i = 10
        return self._wrap[index] if isinstance(index, int) else ImmutableSequenceView[T](self._wrap[index])

    def __iter__(self) -> Iterator[T]:
        if False:
            for i in range(10):
                print('nop')
        return iter(self._wrap)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._wrap)

    def __length_hint__(self) -> int:
        if False:
            print('Hello World!')
        return len(self)

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._wrap)

    def __contains__(self, item: T) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return item in self._wrap

    def index(self, item: T, start: int=0, stop: int=maxsize) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the index of the given item.\n\n        Args:\n            item: The item to find in the sequence.\n            start: Optional start location.\n            stop: Optional stop location.\n\n        Returns:\n            The index of the item in the sequence.\n\n        Raises:\n            ValueError: If the item is not in the sequence.\n        '
        return self._wrap.index(item, start, stop)

    def __reversed__(self) -> Iterator[T]:
        if False:
            return 10
        return reversed(self._wrap)