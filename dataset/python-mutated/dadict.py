"""
Direct Access dictionary.
"""
from scapy.error import Scapy_Exception
from scapy.compat import plain_str
from typing import Any, Dict, Generic, Iterator, List, TypeVar, Union

def fixname(x):
    if False:
        while True:
            i = 10
    '\n    Modifies a string to make sure it can be used as an attribute name.\n    '
    x = plain_str(x)
    if x and str(x[0]) in '0123456789':
        x = 'n_' + x
    return x.translate('________________________________________________0123456789_______ABCDEFGHIJKLMNOPQRSTUVWXYZ______abcdefghijklmnopqrstuvwxyz_____________________________________________________________________________________________________________________________________')

class DADict_Exception(Scapy_Exception):
    pass
_K = TypeVar('_K')
_V = TypeVar('_V')

class DADict(Generic[_K, _V]):
    """
    Direct Access Dictionary

    This acts like a dict, but it provides a direct attribute access
    to its keys through its values. This is used to store protocols,
    manuf...

    For instance, scapy fields will use a DADict as an enum::

        ETHER_TYPES[2048] -> IPv4

    Whereas humans can access::

        ETHER_TYPES.IPv4 -> 2048
    """

    def __init__(self, _name='DADict', **kargs):
        if False:
            return 10
        self._name = _name
        self.d = {}
        self.update(kargs)

    def ident(self, v):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return value that is used as key for the direct access\n        '
        if isinstance(v, (str, bytes)):
            return fixname(v)
        return 'unknown'

    def update(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        for (k, v) in dict(*args, **kwargs).items():
            self[k] = v

    def iterkeys(self):
        if False:
            for i in range(10):
                print('nop')
        for x in self.d:
            if not isinstance(x, str) or x[0] != '_':
                yield x

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.iterkeys())

    def __iter__(self):
        if False:
            return 10
        return self.iterkeys()

    def itervalues(self):
        if False:
            return 10
        return self.d.values()

    def values(self):
        if False:
            return 10
        return list(self.itervalues())

    def _show(self):
        if False:
            for i in range(10):
                print('nop')
        for k in self.iterkeys():
            print('%10s = %r' % (k, self[k]))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s - %s elements>' % (self._name, len(self))

    def __getitem__(self, attr):
        if False:
            while True:
                i = 10
        return self.d[attr]

    def __setitem__(self, attr, val):
        if False:
            while True:
                i = 10
        self.d[attr] = val

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.d)

    def __nonzero__(self):
        if False:
            return 10
        return len(self) > 1
    __bool__ = __nonzero__

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            for (k, v) in self.d.items():
                if self.ident(v) == attr:
                    return k
        raise AttributeError

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.ident(x) for x in self.itervalues()]