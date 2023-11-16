"""Representation for the MongoDB internal MinKey type."""
from __future__ import annotations
from typing import Any

class MinKey:
    """MongoDB internal MinKey type."""
    __slots__ = ()
    _type_marker = 255

    def __getstate__(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return {}

    def __setstate__(self, state: Any) -> None:
        if False:
            return 10
        pass

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        return isinstance(other, MinKey)

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self._type_marker)

    def __ne__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        return not self == other

    def __le__(self, dummy: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def __lt__(self, other: Any) -> bool:
        if False:
            return 10
        return not isinstance(other, MinKey)

    def __ge__(self, other: Any) -> bool:
        if False:
            return 10
        return isinstance(other, MinKey)

    def __gt__(self, dummy: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'MinKey()'