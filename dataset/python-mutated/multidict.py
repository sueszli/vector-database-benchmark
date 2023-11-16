from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Iterator
from collections.abc import MutableMapping
from collections.abc import Sequence
from typing import TypeVar
from mitmproxy.coretypes import serializable
KT = TypeVar('KT')
VT = TypeVar('VT')

class _MultiDict(MutableMapping[KT, VT], metaclass=ABCMeta):
    """
    A MultiDict is a dictionary-like data structure that supports multiple values per key.
    """
    fields: tuple[tuple[KT, VT], ...]
    'The underlying raw datastructure.'

    def __repr__(self):
        if False:
            while True:
                i = 10
        fields = (repr(field) for field in self.fields)
        return '{cls}[{fields}]'.format(cls=type(self).__name__, fields=', '.join(fields))

    @staticmethod
    @abstractmethod
    def _reduce_values(values: Sequence[VT]) -> VT:
        if False:
            return 10
        '\n        If a user accesses multidict["foo"], this method\n        reduces all values for "foo" to a single value that is returned.\n        For example, HTTP headers are folded, whereas we will just take\n        the first cookie we found with that name.\n        '

    @staticmethod
    @abstractmethod
    def _kconv(key: KT) -> KT:
        if False:
            return 10
        '\n        This method converts a key to its canonical representation.\n        For example, HTTP headers are case-insensitive, so this method returns key.lower().\n        '

    def __getitem__(self, key: KT) -> VT:
        if False:
            print('Hello World!')
        values = self.get_all(key)
        if not values:
            raise KeyError(key)
        return self._reduce_values(values)

    def __setitem__(self, key: KT, value: VT) -> None:
        if False:
            return 10
        self.set_all(key, [value])

    def __delitem__(self, key: KT) -> None:
        if False:
            print('Hello World!')
        if key not in self:
            raise KeyError(key)
        key = self._kconv(key)
        self.fields = tuple((field for field in self.fields if key != self._kconv(field[0])))

    def __iter__(self) -> Iterator[KT]:
        if False:
            return 10
        seen = set()
        for (key, _) in self.fields:
            key_kconv = self._kconv(key)
            if key_kconv not in seen:
                seen.add(key_kconv)
                yield key

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len({self._kconv(key) for (key, _) in self.fields})

    def __eq__(self, other) -> bool:
        if False:
            return 10
        if isinstance(other, MultiDict):
            return self.fields == other.fields
        return False

    def get_all(self, key: KT) -> list[VT]:
        if False:
            return 10
        '\n        Return the list of all values for a given key.\n        If that key is not in the MultiDict, the return value will be an empty list.\n        '
        key = self._kconv(key)
        return [value for (k, value) in self.fields if self._kconv(k) == key]

    def set_all(self, key: KT, values: list[VT]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove the old values for a key and add new ones.\n        '
        key_kconv = self._kconv(key)
        new_fields: list[tuple[KT, VT]] = []
        for field in self.fields:
            if self._kconv(field[0]) == key_kconv:
                if values:
                    new_fields.append((field[0], values.pop(0)))
            else:
                new_fields.append(field)
        while values:
            new_fields.append((key, values.pop(0)))
        self.fields = tuple(new_fields)

    def add(self, key: KT, value: VT) -> None:
        if False:
            print('Hello World!')
        '\n        Add an additional value for the given key at the bottom.\n        '
        self.insert(len(self.fields), key, value)

    def insert(self, index: int, key: KT, value: VT) -> None:
        if False:
            while True:
                i = 10
        '\n        Insert an additional value for the given key at the specified position.\n        '
        item = (key, value)
        self.fields = self.fields[:index] + (item,) + self.fields[index:]

    def keys(self, multi: bool=False):
        if False:
            return 10
        '\n        Get all keys.\n\n        If `multi` is True, one key per value will be returned.\n        If `multi` is False, duplicate keys will only be returned once.\n        '
        return (k for (k, _) in self.items(multi))

    def values(self, multi: bool=False):
        if False:
            print('Hello World!')
        '\n        Get all values.\n\n        If `multi` is True, all values will be returned.\n        If `multi` is False, only the first value per key will be returned.\n        '
        return (v for (_, v) in self.items(multi))

    def items(self, multi: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Get all (key, value) tuples.\n\n        If `multi` is True, all `(key, value)` pairs will be returned.\n        If False, only one tuple per key is returned.\n        '
        if multi:
            return self.fields
        else:
            return super().items()

class MultiDict(_MultiDict[KT, VT], serializable.Serializable):
    """A concrete MultiDict, storing its own data."""

    def __init__(self, fields=()):
        if False:
            print('Hello World!')
        super().__init__()
        self.fields = tuple((tuple(i) for i in fields))

    @staticmethod
    def _reduce_values(values):
        if False:
            print('Hello World!')
        return values[0]

    @staticmethod
    def _kconv(key):
        if False:
            for i in range(10):
                print('nop')
        return key

    def get_state(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fields

    def set_state(self, state):
        if False:
            print('Hello World!')
        self.fields = tuple((tuple(x) for x in state))

    @classmethod
    def from_state(cls, state):
        if False:
            while True:
                i = 10
        return cls(state)

class MultiDictView(_MultiDict[KT, VT]):
    """
    The MultiDictView provides the MultiDict interface over calculated data.
    The view itself contains no state - data is retrieved from the parent on
    request, and stored back to the parent on change.
    """

    def __init__(self, getter, setter):
        if False:
            print('Hello World!')
        self._getter = getter
        self._setter = setter
        super().__init__()

    @staticmethod
    def _kconv(key):
        if False:
            return 10
        return key

    @staticmethod
    def _reduce_values(values):
        if False:
            for i in range(10):
                print('nop')
        return values[0]

    @property
    def fields(self):
        if False:
            while True:
                i = 10
        return self._getter()

    @fields.setter
    def fields(self, value):
        if False:
            print('Hello World!')
        self._setter(value)

    def copy(self) -> 'MultiDict[KT,VT]':
        if False:
            while True:
                i = 10
        return MultiDict(self.fields)