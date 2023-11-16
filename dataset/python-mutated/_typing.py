from typing import TypeVar, Protocol
T = TypeVar('T')

class RingElement(Protocol):
    """A ring element.

    Must support ``+``, ``-``, ``*``, ``**`` and ``-``.
    """

    def __add__(self: T, other: T, /) -> T:
        if False:
            while True:
                i = 10
        ...

    def __sub__(self: T, other: T, /) -> T:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __mul__(self: T, other: T, /) -> T:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __pow__(self: T, other: int, /) -> T:
        if False:
            return 10
        ...

    def __neg__(self: T, /) -> T:
        if False:
            i = 10
            return i + 15
        ...