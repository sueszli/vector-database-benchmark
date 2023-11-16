from __future__ import annotations
from typing import Generic, TypeVar
Key = TypeVar('Key')
Value = TypeVar('Value')

class TwoWayDict(Generic[Key, Value]):
    """
    A two-way mapping offering O(1) access in both directions.

    Wraps two dictionaries and uses them to provide efficient access to
    both values (given keys) and keys (given values).
    """

    def __init__(self, initial: dict[Key, Value]) -> None:
        if False:
            while True:
                i = 10
        self._forward: dict[Key, Value] = initial
        self._reverse: dict[Value, Key] = {value: key for (key, value) in initial.items()}

    def __setitem__(self, key: Key, value: Value) -> None:
        if False:
            while True:
                i = 10
        self._forward.__setitem__(key, value)
        self._reverse.__setitem__(value, key)

    def __delitem__(self, key: Key) -> None:
        if False:
            return 10
        value = self._forward[key]
        self._forward.__delitem__(key)
        self._reverse.__delitem__(value)

    def __iter__(self):
        if False:
            return 10
        return iter(self._forward)

    def get(self, key: Key) -> Value:
        if False:
            i = 10
            return i + 15
        'Given a key, efficiently lookup and return the associated value.\n\n        Args:\n            key: The key\n\n        Returns:\n            The value\n        '
        return self._forward.get(key)

    def get_key(self, value: Value) -> Key:
        if False:
            for i in range(10):
                print('nop')
        'Given a value, efficiently lookup and return the associated key.\n\n        Args:\n            value: The value\n\n        Returns:\n            The key\n        '
        return self._reverse.get(value)

    def contains_value(self, value: Value) -> bool:
        if False:
            return 10
        'Check if `value` is a value within this TwoWayDict.\n\n        Args:\n            value: The value to check.\n\n        Returns:\n            True if the value is within the values of this dict.\n        '
        return value in self._reverse

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._forward)

    def __contains__(self, item: Key) -> bool:
        if False:
            print('Hello World!')
        return item in self._forward