"""Representation for the MongoDB internal MaxKey type."""
from __future__ import annotations
from typing import Any

class MaxKey:
    """MongoDB internal MaxKey type."""
    __slots__ = ()
    _type_marker = 127

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
            return 10
        return isinstance(other, MaxKey)

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self._type_marker)

    def __ne__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        return not self == other

    def __le__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, MaxKey)

    def __lt__(self, dummy: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def __ge__(self, dummy: Any) -> bool:
        if False:
            while True:
                i = 10
        return True

    def __gt__(self, other: Any) -> bool:
        if False:
            return 10
        return not isinstance(other, MaxKey)

    def __repr__(self) -> str:
        if False:
            return 10
        return 'MaxKey()'