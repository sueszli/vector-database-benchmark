from __future__ import annotations
import weakref
from weakref import ref
from _weakrefset import _IterationGuard
from collections.abc import MutableMapping, Mapping
from torch import Tensor
import collections.abc as _collections_abc
WeakRef = ref
__all__ = ['TensorWeakRef', 'WeakIdRef', 'WeakIdKeyDictionary', 'WeakTensorKeyDictionary']

class WeakIdRef(weakref.ref):
    __slots__ = ['_id']

    def __init__(self, key, callback=None):
        if False:
            i = 10
            return i + 15
        self._id = id(key)
        super().__init__(key, callback)

    def __call__(self):
        if False:
            return 10
        r = super().__call__()
        if hasattr(r, '_fix_weakref'):
            r._fix_weakref()
        return r

    def __hash__(self):
        if False:
            return 10
        return self._id

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other

class WeakIdKeyDictionary(MutableMapping):
    data: dict[WeakIdRef, object]

    def __init__(self, dict=None):
        if False:
            while True:
                i = 10
        self.data = {}

        def remove(k, selfref=ref(self)):
            if False:
                print('Hello World!')
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass
        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self):
        if False:
            while True:
                i = 10
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return
            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self):
        if False:
            while True:
                i = 10
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        self._dirty_len = True
        del self.data[WeakIdRef(key)]

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.data[WeakIdRef(key)]

    def __len__(self):
        if False:
            return 10
        if self._dirty_len and self._pending_removals:
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<{self.__class__.__name__} at {id(self):#x}>'

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        self.data[WeakIdRef(key, self._remove)] = value

    def copy(self):
        if False:
            i = 10
            return i + 15
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for (key, value) in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new
    __copy__ = copy

    def __deepcopy__(self, memo):
        if False:
            return 10
        from copy import deepcopy
        new = self.__class__()
        with _IterationGuard(self):
            for (key, value) in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new

    def get(self, key, default=None):
        if False:
            while True:
                i = 10
        return self.data.get(WeakIdRef(key), default)

    def __contains__(self, key):
        if False:
            print('Hello World!')
        try:
            wr = WeakIdRef(key)
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        if False:
            return 10
        with _IterationGuard(self):
            for (wr, value) in self.data.items():
                key = wr()
                if key is not None:
                    yield (key, value)

    def keys(self):
        if False:
            while True:
                i = 10
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj
    __iter__ = keys

    def values(self):
        if False:
            while True:
                i = 10
        with _IterationGuard(self):
            for (wr, value) in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a list of weak references to the keys.\n\n        The references are not guaranteed to be 'live' at the time\n        they are used, so the result of calling the references needs\n        to be checked before being used.  This can be used to avoid\n        creating references that will cause the garbage collector to\n        keep the keys around longer than needed.\n\n        "
        return list(self.data)

    def popitem(self):
        if False:
            while True:
                i = 10
        self._dirty_len = True
        while True:
            (key, value) = self.data.popitem()
            o = key()
            if o is not None:
                return (o, value)

    def pop(self, key, *args):
        if False:
            while True:
                i = 10
        self._dirty_len = True
        return self.data.pop(WeakIdRef(key), *args)

    def setdefault(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        return self.data.setdefault(WeakIdRef(key, self._remove), default)

    def update(self, dict=None, **kwargs):
        if False:
            print('Hello World!')
        d = self.data
        if dict is not None:
            if not hasattr(dict, 'items'):
                dict = type({})(dict)
            for (key, value) in dict.items():
                d[WeakIdRef(key, self._remove)] = value
        if len(kwargs):
            self.update(kwargs)

    def __ior__(self, other):
        if False:
            return 10
        self.update(other)
        return self

    def __or__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for (k, v) in self.items()} == {id(k): v for (k, v) in other.items()}
WeakTensorKeyDictionary = WeakIdKeyDictionary

class TensorWeakRef:
    """Wrapper around a weak ref of a Tensor that handles the _fix_weakref() call required when unwrapping a Tensor weakref."""
    ref: WeakRef[Tensor]

    def __init__(self, tensor: Tensor):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(tensor, Tensor)
        self.ref = weakref.ref(tensor)

    def __call__(self):
        if False:
            i = 10
            return i + 15
        out = self.ref()
        if out is None:
            return out
        assert isinstance(out, Tensor)
        out._fix_weakref()
        return out