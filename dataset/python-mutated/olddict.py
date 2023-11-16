"""
A dict subclass for Python 3 that behaves like Python 2's dict

Example use:

>>> from past.builtins import dict
>>> d1 = dict()    # instead of {} for an empty dict
>>> d2 = dict(key1='value1', key2='value2')

The keys, values and items methods now return lists on Python 3.x and there are
methods for iterkeys, itervalues, iteritems, and viewkeys etc.

>>> for d in (d1, d2):
...     assert isinstance(d.keys(), list)
...     assert isinstance(d.values(), list)
...     assert isinstance(d.items(), list)
"""
import sys
from past.utils import with_metaclass
_builtin_dict = dict
ver = sys.version_info[:2]

class BaseOldDict(type):

    def __instancecheck__(cls, instance):
        if False:
            i = 10
            return i + 15
        return isinstance(instance, _builtin_dict)

class olddict(with_metaclass(BaseOldDict, _builtin_dict)):
    """
    A backport of the Python 3 dict object to Py2
    """
    iterkeys = _builtin_dict.keys
    viewkeys = _builtin_dict.keys

    def keys(self):
        if False:
            return 10
        return list(super(olddict, self).keys())
    itervalues = _builtin_dict.values
    viewvalues = _builtin_dict.values

    def values(self):
        if False:
            for i in range(10):
                print('nop')
        return list(super(olddict, self).values())
    iteritems = _builtin_dict.items
    viewitems = _builtin_dict.items

    def items(self):
        if False:
            while True:
                i = 10
        return list(super(olddict, self).items())

    def has_key(self, k):
        if False:
            i = 10
            return i + 15
        '\n        D.has_key(k) -> True if D has a key k, else False\n        '
        return k in self

    def __native__(self):
        if False:
            print('Hello World!')
        '\n        Hook for the past.utils.native() function\n        '
        return super(oldbytes, self)
__all__ = ['olddict']