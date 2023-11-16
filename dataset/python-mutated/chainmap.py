from collections import MutableMapping

class ChainMap(MutableMapping):
    """ A ChainMap groups multiple dicts (or other mappings) together
    to create a single, updateable view.

    The underlying mappings are stored in a list.  That list is public and can
    be accessed or updated using the *maps* attribute.  There is no other
    state.

    Lookups search the underlying mappings successively until a key is found.
    In contrast, writes, updates, and deletions only operate on the first
    mapping.

    """

    def __init__(self, *maps):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a ChainMap by setting *maps* to the given mappings.\n        If no mappings are provided, a single empty dictionary is used.\n\n        '
        self.maps = list(maps) or [{}]

    def __missing__(self, key):
        if False:
            i = 10
            return i + 15
        raise KeyError(key)

    def __getitem__(self, key):
        if False:
            return 10
        for mapping in self.maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        return self.__missing__(key)

    def get(self, key, default=None):
        if False:
            print('Hello World!')
        return self[key] if key in self else default

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(set().union(*self.maps))

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(set().union(*self.maps))

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        return any((key in m for m in self.maps))

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return any(self.maps)

    def __repr__(self):
        if False:
            return 10
        return '{0.__class__.__name__}({1})'.format(self, ', '.join(map(repr, self.maps)))

    @classmethod
    def fromkeys(cls, iterable, *args):
        if False:
            for i in range(10):
                print('nop')
        'Create a ChainMap with a single dict created from the iterable.'
        return cls(dict.fromkeys(iterable, *args))

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'New ChainMap or subclass with a new copy of maps[0] and refs to maps[1:]'
        return self.__class__(self.maps[0].copy(), *self.maps[1:])
    __copy__ = copy

    def new_child(self, m=None):
        if False:
            i = 10
            return i + 15
        'New ChainMap with a new map followed by all previous maps.\n        If no map is provided, an empty dict is used.\n        '
        if m is None:
            m = {}
        return self.__class__(m, *self.maps)

    @property
    def parents(self):
        if False:
            i = 10
            return i + 15
        'New ChainMap from maps[1:].'
        return self.__class__(*self.maps[1:])

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        self.maps[0][key] = value

    def __delitem__(self, key):
        if False:
            while True:
                i = 10
        try:
            del self.maps[0][key]
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def popitem(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove and return an item pair from maps[0]. Raise KeyError is maps[0] is empty.'
        try:
            return self.maps[0].popitem()
        except KeyError:
            raise KeyError('No keys found in the first mapping.')

    def pop(self, key, *args):
        if False:
            for i in range(10):
                print('nop')
        'Remove *key* from maps[0] and return its value. Raise KeyError if *key* not in maps[0].'
        try:
            return self.maps[0].pop(key, *args)
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear maps[0], leaving maps[1:] intact.'
        self.maps[0].clear()