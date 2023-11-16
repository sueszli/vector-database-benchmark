from operator import eq, lt, le, gt, ge
from .robottypes import type_name

class Sortable:
    """Base class for sorting based self._sort_key"""
    _sort_key = NotImplemented

    def __test(self, operator, other, require_sortable=True):
        if False:
            i = 10
            return i + 15
        if isinstance(other, Sortable):
            return operator(self._sort_key, other._sort_key)
        if not require_sortable:
            return False
        raise TypeError("Cannot sort '%s' and '%s'." % (type_name(self), type_name(other)))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.__test(eq, other, require_sortable=False)

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__test(lt, other)

    def __le__(self, other):
        if False:
            print('Hello World!')
        return self.__test(le, other)

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__test(gt, other)

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__test(ge, other)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self._sort_key)