"""
A dict subclass for Python 2 that behaves like Python 3's dict

Example use:

>>> from builtins import dict
>>> d1 = dict()    # instead of {} for an empty dict
>>> d2 = dict(key1='value1', key2='value2')

The keys, values and items methods now return iterators on Python 2.x
(with set-like behaviour on Python 2.7).

>>> for d in (d1, d2):
...     assert not isinstance(d.keys(), list)
...     assert not isinstance(d.values(), list)
...     assert not isinstance(d.items(), list)
"""
import sys
from future.utils import with_metaclass
from future.types.newobject import newobject
_builtin_dict = dict
ver = sys.version_info

class BaseNewDict(type):

    def __instancecheck__(cls, instance):
        if False:
            i = 10
            return i + 15
        if cls == newdict:
            return isinstance(instance, _builtin_dict)
        else:
            return issubclass(instance.__class__, cls)

class newdict(with_metaclass(BaseNewDict, _builtin_dict)):
    """
    A backport of the Python 3 dict object to Py2
    """
    if ver >= (3,):
        pass
    elif ver >= (2, 7):
        items = dict.viewitems
        keys = dict.viewkeys
        values = dict.viewvalues
    else:
        items = dict.iteritems
        keys = dict.iterkeys
        values = dict.itervalues

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        dict() -> new empty dictionary\n        dict(mapping) -> new dictionary initialized from a mapping object's\n            (key, value) pairs\n        dict(iterable) -> new dictionary initialized as if via:\n            d = {}\n            for k, v in iterable:\n                d[k] = v\n        dict(**kwargs) -> new dictionary initialized with the name=value pairs\n            in the keyword argument list.  For example:  dict(one=1, two=2)\n        "
        return super(newdict, cls).__new__(cls, *args)

    def __native__(self):
        if False:
            print('Hello World!')
        '\n        Hook for the future.utils.native() function\n        '
        return dict(self)
__all__ = ['newdict']