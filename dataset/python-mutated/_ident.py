from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from weakref import WeakKeyDictionary
from weakref import ref
from heapq import heappop
from heapq import heappush
__all__ = ['IdentRegistry']

class ValuedWeakRef(ref):
    """
    A weak ref with an associated value.
    """
    __slots__ = ('value',)

class IdentRegistry(object):
    """
    Maintains a unique mapping of (small) non-negative integer identifiers
    to objects that can be weakly referenced.

    It is guaranteed that no two objects will have the the same
    identifier at the same time, as long as those objects are
    also uniquely hashable.
    """

    def __init__(self):
        if False:
            return 10
        self._registry = WeakKeyDictionary()
        self._available_idents = []

    def get_ident(self, obj):
        if False:
            return 10
        '\n        Retrieve the identifier for *obj*, creating one\n        if necessary.\n        '
        try:
            return self._registry[obj][0]
        except KeyError:
            pass
        if self._available_idents:
            ident = heappop(self._available_idents)
        else:
            ident = len(self._registry)
        vref = ValuedWeakRef(obj, self._return_ident)
        vref.value = ident
        self._registry[obj] = (ident, vref)
        return ident

    def _return_ident(self, vref):
        if False:
            return 10
        if heappush is not None:
            heappush(self._available_idents, vref.value)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._registry)
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__ident')