"""
Pure-Python implementation of a Python 2-like str object for Python 3.
"""
from numbers import Integral
from past.utils import PY2, with_metaclass
if PY2:
    from collections import Iterable
else:
    from collections.abc import Iterable
_builtin_bytes = bytes

class BaseOldStr(type):

    def __instancecheck__(cls, instance):
        if False:
            while True:
                i = 10
        return isinstance(instance, _builtin_bytes)

def unescape(s):
    if False:
        return 10
    "\n    Interprets strings with escape sequences\n\n    Example:\n    >>> s = unescape(r'abc\\\\def')   # i.e. 'abc\\\\\\\\def'\n    >>> print(s)\n    'abc\\def'\n    >>> s2 = unescape('abc\\\\ndef')\n    >>> len(s2)\n    8\n    >>> print(s2)\n    abc\n    def\n    "
    return s.encode().decode('unicode_escape')

class oldstr(with_metaclass(BaseOldStr, _builtin_bytes)):
    """
    A forward port of the Python 2 8-bit string object to Py3
    """

    @property
    def __iter__(self):
        if False:
            print('Hello World!')
        raise AttributeError

    def __dir__(self):
        if False:
            while True:
                i = 10
        return [thing for thing in dir(_builtin_bytes) if thing != '__iter__']

    def __repr__(self):
        if False:
            return 10
        s = super(oldstr, self).__repr__()
        return s[1:]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        s = super(oldstr, self).__str__()
        assert s[:2] == "b'" and s[-1] == "'"
        return unescape(s[2:-1])

    def __getitem__(self, y):
        if False:
            while True:
                i = 10
        if isinstance(y, Integral):
            return super(oldstr, self).__getitem__(slice(y, y + 1))
        else:
            return super(oldstr, self).__getitem__(y)

    def __getslice__(self, *args):
        if False:
            while True:
                i = 10
        return self.__getitem__(slice(*args))

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        if isinstance(key, int):
            return False

    def __native__(self):
        if False:
            i = 10
            return i + 15
        return bytes(self)
__all__ = ['oldstr']