""" This module is only an abstraction of OrderedDict as present in 2.7 and 3.x.

It is not in 2.6, for this version we are using the odict.py as mentioned in the
PEP-0372.

This can be removed safely after Python2.6 support is dropped (if ever), note
that the documentation was removed, as it's not interesting really, being
redundant to the Python 2.7 documentation.

Starting with Python 3.6, we can safely use the built-in dictionary.
"""
from nuitka.PythonVersions import python_version
try:
    if python_version >= 864:
        OrderedDict = dict
    else:
        from collections import OrderedDict
except ImportError:
    assert python_version < 624
    from copy import deepcopy
    from itertools import imap, izip
    missing = object()

    class OrderedDict(dict):

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            dict.__init__(self)
            self._keys = []
            self.update(*args, **kwargs)

        def __delitem__(self, key):
            if False:
                print('Hello World!')
            dict.__delitem__(self, key)
            self._keys.remove(key)

        def __setitem__(self, key, item):
            if False:
                while True:
                    i = 10
            if key not in self:
                self._keys.append(key)
            dict.__setitem__(self, key, item)

        def __deepcopy__(self, memo=None):
            if False:
                return 10
            if memo is None:
                memo = {}
            d = memo.get(id(self), missing)
            if d is not missing:
                return d
            memo[id(self)] = d = self.__class__()
            dict.__init__(d, deepcopy(self.items(), memo))
            d._keys = self._keys[:]
            return d

        def __getstate__(self):
            if False:
                while True:
                    i = 10
            return {'items': dict(self), 'keys': self._keys}

        def __setstate__(self, d):
            if False:
                i = 10
                return i + 15
            self._keys = d['keys']
            dict.update(d['items'])

        def __reversed__(self):
            if False:
                i = 10
                return i + 15
            return reversed(self._keys)

        def __eq__(self, other):
            if False:
                print('Hello World!')
            if isinstance(other, OrderedDict):
                if not dict.__eq__(self, other):
                    return False
                return self.items() == other.items()
            return dict.__eq__(self, other)

        def __ne__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            return not self.__eq__(other)

        def __cmp__(self, other):
            if False:
                while True:
                    i = 10
            if isinstance(other, OrderedDict):
                return cmp(self.items(), other.items())
            elif isinstance(other, dict):
                return dict.__cmp__(self, other)
            return NotImplemented

        @classmethod
        def fromkeys(cls, iterable, default=None):
            if False:
                print('Hello World!')
            return cls(((key, default) for key in iterable))

        def clear(self):
            if False:
                i = 10
                return i + 15
            del self._keys[:]
            dict.clear(self)

        def copy(self):
            if False:
                i = 10
                return i + 15
            return self.__class__(self)

        def items(self):
            if False:
                return 10
            return zip(self._keys, self.values())

        def iteritems(self):
            if False:
                i = 10
                return i + 15
            return izip(self._keys, self.itervalues())

        def keys(self):
            if False:
                print('Hello World!')
            return self._keys[:]

        def iterkeys(self):
            if False:
                print('Hello World!')
            return iter(self._keys)

        def pop(self, key, default=missing):
            if False:
                print('Hello World!')
            if default is missing:
                return dict.pop(self, key)
            elif key not in self:
                return default
            self._keys.remove(key)
            return dict.pop(self, key, default)

        def popitem(self, key):
            if False:
                print('Hello World!')
            self._keys.remove(key)
            return dict.popitem(key)

        def setdefault(self, key, default=None):
            if False:
                print('Hello World!')
            if key not in self:
                self._keys.append(key)
            dict.setdefault(self, key, default)

        def update(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            sources = []
            if len(args) == 1:
                if hasattr(args[0], 'iteritems'):
                    sources.append(args[0].iteritems())
                else:
                    sources.append(iter(args[0]))
            elif args:
                raise TypeError('expected at most one positional argument')
            if kwargs:
                sources.append(kwargs.iteritems())
            for iterable in sources:
                for (key, val) in iterable:
                    self[key] = val

        def values(self):
            if False:
                print('Hello World!')
            return map(self.get, self._keys)

        def itervalues(self):
            if False:
                print('Hello World!')
            return imap(self.get, self._keys)

        def index(self, item):
            if False:
                return 10
            return self._keys.index(item)

        def byindex(self, item):
            if False:
                i = 10
                return i + 15
            key = self._keys[item]
            return (key, dict.__getitem__(self, key))

        def reverse(self):
            if False:
                print('Hello World!')
            self._keys.reverse()

        def sort(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self._keys.sort(*args, **kwargs)

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return 'OrderedDict(%r)' % self.items()
        __copy__ = copy
        __iter__ = iterkeys