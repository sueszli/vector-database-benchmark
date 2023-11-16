"""DNS name dictionary"""
from collections.abc import MutableMapping
import dns.name

class NameDict(MutableMapping):
    """A dictionary whose keys are dns.name.Name objects.

    In addition to being like a regular Python dictionary, this
    dictionary can also get the deepest match for a given key.
    """
    __slots__ = ['max_depth', 'max_depth_items', '__store']

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.__store = dict()
        self.max_depth = 0
        self.max_depth_items = 0
        self.update(dict(*args, **kwargs))

    def __update_max_depth(self, key):
        if False:
            for i in range(10):
                print('nop')
        if len(key) == self.max_depth:
            self.max_depth_items = self.max_depth_items + 1
        elif len(key) > self.max_depth:
            self.max_depth = len(key)
            self.max_depth_items = 1

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self.__store[key]

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(key, dns.name.Name):
            raise ValueError('NameDict key must be a name')
        self.__store[key] = value
        self.__update_max_depth(key)

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        self.__store.pop(key)
        if len(key) == self.max_depth:
            self.max_depth_items = self.max_depth_items - 1
        if self.max_depth_items == 0:
            self.max_depth = 0
            for k in self.__store:
                self.__update_max_depth(k)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.__store)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.__store)

    def has_key(self, key):
        if False:
            i = 10
            return i + 15
        return key in self.__store

    def get_deepest_match(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Find the deepest match to *name* in the dictionary.\n\n        The deepest match is the longest name in the dictionary which is\n        a superdomain of *name*.  Note that *superdomain* includes matching\n        *name* itself.\n\n        *name*, a ``dns.name.Name``, the name to find.\n\n        Returns a ``(key, value)`` where *key* is the deepest\n        ``dns.name.Name``, and *value* is the value associated with *key*.\n        '
        depth = len(name)
        if depth > self.max_depth:
            depth = self.max_depth
        for i in range(-depth, 0):
            n = dns.name.Name(name[i:])
            if n in self:
                return (n, self[n])
        v = self[dns.name.empty]
        return (dns.name.empty, v)