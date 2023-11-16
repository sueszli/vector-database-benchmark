"""
A list subclass for Python 2 that behaves like Python 3's list.

The primary difference is that lists have a .copy() method in Py3.

Example use:

>>> from builtins import list
>>> l1 = list()    # instead of {} for an empty list
>>> l1.append('hello')
>>> l2 = l1.copy()

"""
import sys
import copy
from future.utils import with_metaclass
from future.types.newobject import newobject
_builtin_list = list
ver = sys.version_info[:2]

class BaseNewList(type):

    def __instancecheck__(cls, instance):
        if False:
            return 10
        if cls == newlist:
            return isinstance(instance, _builtin_list)
        else:
            return issubclass(instance.__class__, cls)

class newlist(with_metaclass(BaseNewList, _builtin_list)):
    """
    A backport of the Python 3 list object to Py2
    """

    def copy(self):
        if False:
            i = 10
            return i + 15
        '\n        L.copy() -> list -- a shallow copy of L\n        '
        return copy.copy(self)

    def clear(self):
        if False:
            print('Hello World!')
        'L.clear() -> None -- remove all items from L'
        for i in range(len(self)):
            self.pop()

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        "\n        list() -> new empty list\n        list(iterable) -> new list initialized from iterable's items\n        "
        if len(args) == 0:
            return super(newlist, cls).__new__(cls)
        elif type(args[0]) == newlist:
            value = args[0]
        else:
            value = args[0]
        return super(newlist, cls).__new__(cls, value)

    def __add__(self, value):
        if False:
            return 10
        return newlist(super(newlist, self).__add__(value))

    def __radd__(self, left):
        if False:
            i = 10
            return i + 15
        ' left + self '
        try:
            return newlist(left) + self
        except:
            return NotImplemented

    def __getitem__(self, y):
        if False:
            while True:
                i = 10
        '\n        x.__getitem__(y) <==> x[y]\n\n        Warning: a bug in Python 2.x prevents indexing via a slice from\n        returning a newlist object.\n        '
        if isinstance(y, slice):
            return newlist(super(newlist, self).__getitem__(y))
        else:
            return super(newlist, self).__getitem__(y)

    def __native__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hook for the future.utils.native() function\n        '
        return list(self)

    def __nonzero__(self):
        if False:
            return 10
        return len(self) > 0
__all__ = ['newlist']