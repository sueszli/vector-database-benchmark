import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, MutableMapping, TypeVar
V = TypeVar('V')
Self = TypeVar('Self', bound='NormalizedDict')

def normalize(string: str, ignore: 'Sequence[str]'=(), caseless: bool=True, spaceless: bool=True) -> str:
    if False:
        return 10
    'Normalize the ``string`` according to the given spec.\n\n    By default, string is turned to lower case (actually case-folded) and all\n    whitespace is removed. Additional characters can be removed by giving them\n    in ``ignore`` list.\n    '
    if spaceless:
        string = ''.join(string.split())
    if caseless:
        string = string.casefold()
        ignore = [i.casefold() for i in ignore]
    if ignore:
        for ign in ignore:
            if ign in string:
                string = string.replace(ign, '')
    return string

def normalize_whitespace(string):
    if False:
        print('Hello World!')
    return re.sub('\\s', ' ', string, flags=re.UNICODE)

class NormalizedDict(MutableMapping[str, V]):
    """Custom dictionary implementation automatically normalizing keys."""

    def __init__(self, initial: 'Mapping[str, V]|Iterable[tuple[str, V]]|None'=None, ignore: 'Sequence[str]'=(), caseless: bool=True, spaceless: bool=True):
        if False:
            for i in range(10):
                print('nop')
        'Initialized with possible initial value and normalizing spec.\n\n        Initial values can be either a dictionary or an iterable of name/value\n        pairs.\n\n        Normalizing spec has exact same semantics as with the :func:`normalize`\n        function.\n        '
        self._data: 'dict[str, V]' = {}
        self._keys: 'dict[str, str]' = {}
        self._normalize = lambda s: normalize(s, ignore, caseless, spaceless)
        if initial:
            self.update(initial)

    @property
    def normalized_keys(self) -> 'tuple[str, ...]':
        if False:
            i = 10
            return i + 15
        return tuple(self._keys)

    def __getitem__(self, key: str) -> V:
        if False:
            i = 10
            return i + 15
        return self._data[self._normalize(key)]

    def __setitem__(self, key: str, value: V):
        if False:
            for i in range(10):
                print('nop')
        norm_key = self._normalize(key)
        self._data[norm_key] = value
        self._keys.setdefault(norm_key, key)

    def __delitem__(self, key: str):
        if False:
            while True:
                i = 10
        norm_key = self._normalize(key)
        del self._data[norm_key]
        del self._keys[norm_key]

    def __iter__(self) -> 'Iterator[str]':
        if False:
            i = 10
            return i + 15
        return (self._keys[norm_key] for norm_key in sorted(self._keys))

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self._data)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        items = ', '.join((f'{key!r}: {self[key]!r}' for key in self))
        return f'{{{items}}}'

    def __repr__(self) -> str:
        if False:
            return 10
        name = type(self).__name__
        params = str(self) if self else ''
        return f'{name}({params})'

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, Mapping):
            return False
        if not isinstance(other, NormalizedDict):
            other = NormalizedDict(other)
        return self._data == other._data

    def copy(self: Self) -> Self:
        if False:
            while True:
                i = 10
        copy = type(self)()
        copy._data = self._data.copy()
        copy._keys = self._keys.copy()
        copy._normalize = self._normalize
        return copy

    def __contains__(self, key: str) -> bool:
        if False:
            i = 10
            return i + 15
        return self._normalize(key) in self._data

    def clear(self):
        if False:
            print('Hello World!')
        self._data.clear()
        self._keys.clear()