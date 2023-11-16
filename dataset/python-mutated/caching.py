from __future__ import annotations
import functools
import weakref
from collections import Counter, defaultdict
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Callable
from bidict import bidict
from ibis.common.exceptions import IbisError
if TYPE_CHECKING:
    from collections.abc import Iterator

def memoize(func: Callable) -> Callable:
    if False:
        print('Hello World!')
    'Memoize a function.'
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        key = (args, tuple(kwargs.items()))
        try:
            return cache[key]
        except KeyError:
            result = func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper

class WeakCache(MutableMapping):
    __slots__ = ('_data',)
    _data: dict

    def __init__(self):
        if False:
            return 10
        object.__setattr__(self, '_data', {})

    def __setattr__(self, name, value):
        if False:
            i = 10
            return i + 15
        raise TypeError(f"can't set {name}")

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        if False:
            i = 10
            return i + 15
        return iter(self._data)

    def __setitem__(self, key, value) -> None:
        if False:
            while True:
                i = 10
        identifiers = tuple((id(item) for item in key))

        def callback(ref_):
            if False:
                return 10
            return self._data.pop(identifiers, None)
        refs = tuple((weakref.ref(item, callback) for item in key))
        self._data[identifiers] = (value, refs)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        identifiers = tuple((id(item) for item in key))
        (value, _) = self._data[identifiers]
        return value

    def __delitem__(self, key):
        if False:
            while True:
                i = 10
        identifiers = tuple((id(item) for item in key))
        del self._data[identifiers]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}({self._data})'

class RefCountedCache:
    """A cache with reference-counted keys.

    We could implement `MutableMapping`, but the `__setitem__` implementation
    doesn't make sense and the `len` and `__iter__` methods aren't used.

    We can implement that interface if and when we need to.
    """

    def __init__(self, *, populate: Callable[[str, Any], None], lookup: Callable[[str], Any], finalize: Callable[[Any], None], generate_name: Callable[[], str], key: Callable[[Any], Any]) -> None:
        if False:
            return 10
        self.cache = bidict()
        self.refs: Counter = Counter()
        self.populate = populate
        self.lookup = lookup
        self.finalize = finalize
        self.names: defaultdict = defaultdict(generate_name)
        self.key = key or (lambda x: x)

    def get(self, key, default=None):
        if False:
            return 10
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        result = self.cache[key]
        self.refs[key] += 1
        return result

    def store(self, input) -> None:
        if False:
            return 10
        'Compute and store a reference to `key`.'
        key = self.key(input)
        name = self.names[key]
        self.populate(name, input)
        self.cache[key] = self.lookup(name)
        self.refs[key] = 0

    def __delitem__(self, key) -> None:
        if False:
            return 10
        if (inv_key := self.cache.inverse.get(key)) is None:
            raise IbisError('Key has already been released. Did you call `.release()` twice on the same expression?')
        self.refs[inv_key] -= 1
        assert self.refs[inv_key] >= 0, f'refcount is negative: {self.refs[inv_key]:d}'
        if not self.refs[inv_key]:
            del self.cache[inv_key], self.refs[inv_key]
            self.finalize(key)