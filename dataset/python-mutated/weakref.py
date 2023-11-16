"""Weak reference support for Python.

This module is an implementation of PEP 205:

https://www.python.org/dev/peps/pep-0205/
"""
from _weakref import getweakrefcount, getweakrefs, ref, proxy, CallableProxyType, ProxyType, ReferenceType, _remove_dead_weakref
from _weakrefset import WeakSet, _IterationGuard
import _collections_abc
import sys
import itertools
ProxyTypes = (ProxyType, CallableProxyType)
__all__ = ['ref', 'proxy', 'getweakrefcount', 'getweakrefs', 'WeakKeyDictionary', 'ReferenceType', 'ProxyType', 'CallableProxyType', 'ProxyTypes', 'WeakValueDictionary', 'WeakSet', 'WeakMethod', 'finalize']
_collections_abc.Set.register(WeakSet)
_collections_abc.MutableSet.register(WeakSet)

class WeakMethod(ref):
    """
    A custom `weakref.ref` subclass which simulates a weak reference to
    a bound method, working around the lifetime problem of bound methods.
    """
    __slots__ = ('_func_ref', '_meth_type', '_alive', '__weakref__')

    def __new__(cls, meth, callback=None):
        if False:
            print('Hello World!')
        try:
            obj = meth.__self__
            func = meth.__func__
        except AttributeError:
            raise TypeError('argument should be a bound method, not {}'.format(type(meth))) from None

        def _cb(arg):
            if False:
                while True:
                    i = 10
            self = self_wr()
            if self._alive:
                self._alive = False
                if callback is not None:
                    callback(self)
        self = ref.__new__(cls, obj, _cb)
        self._func_ref = ref(func, _cb)
        self._meth_type = type(meth)
        self._alive = True
        self_wr = ref(self)
        return self

    def __call__(self):
        if False:
            print('Hello World!')
        obj = super().__call__()
        func = self._func_ref()
        if obj is None or func is None:
            return None
        return self._meth_type(func, obj)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, WeakMethod):
            if not self._alive or not other._alive:
                return self is other
            return ref.__eq__(self, other) and self._func_ref == other._func_ref
        return NotImplemented

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, WeakMethod):
            if not self._alive or not other._alive:
                return self is not other
            return ref.__ne__(self, other) or self._func_ref != other._func_ref
        return NotImplemented
    __hash__ = ref.__hash__

class WeakValueDictionary(_collections_abc.MutableMapping):
    """Mapping class that references values weakly.

    Entries in the dictionary will be discarded when no strong
    reference to the value exists anymore
    """

    def __init__(self, other=(), /, **kw):
        if False:
            for i in range(10):
                print('nop')

        def remove(wr, selfref=ref(self), _atomic_removal=_remove_dead_weakref):
            if False:
                print('Hello World!')
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(wr.key)
                else:
                    _atomic_removal(self.data, wr.key)
        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self.data = {}
        self.update(other, **kw)

    def _commit_removals(self, _atomic_removal=_remove_dead_weakref):
        if False:
            print('Hello World!')
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return
            _atomic_removal(d, key)

    def __getitem__(self, key):
        if False:
            return 10
        if self._pending_removals:
            self._commit_removals()
        o = self.data[key]()
        if o is None:
            raise KeyError(key)
        else:
            return o

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        if self._pending_removals:
            self._commit_removals()
        del self.data[key]

    def __len__(self):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        return len(self.data)

    def __contains__(self, key):
        if False:
            return 10
        if self._pending_removals:
            self._commit_removals()
        try:
            o = self.data[key]()
        except KeyError:
            return False
        return o is not None

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s at %#x>' % (self.__class__.__name__, id(self))

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        self.data[key] = KeyedRef(value, self._remove, key)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        if self._pending_removals:
            self._commit_removals()
        new = WeakValueDictionary()
        with _IterationGuard(self):
            for (key, wr) in self.data.items():
                o = wr()
                if o is not None:
                    new[key] = o
        return new
    __copy__ = copy

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        from copy import deepcopy
        if self._pending_removals:
            self._commit_removals()
        new = self.__class__()
        with _IterationGuard(self):
            for (key, wr) in self.data.items():
                o = wr()
                if o is not None:
                    new[deepcopy(key, memo)] = o
        return new

    def get(self, key, default=None):
        if False:
            i = 10
            return i + 15
        if self._pending_removals:
            self._commit_removals()
        try:
            wr = self.data[key]
        except KeyError:
            return default
        else:
            o = wr()
            if o is None:
                return default
            else:
                return o

    def items(self):
        if False:
            return 10
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            for (k, wr) in self.data.items():
                v = wr()
                if v is not None:
                    yield (k, v)

    def keys(self):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            for (k, wr) in self.data.items():
                if wr() is not None:
                    yield k
    __iter__ = keys

    def itervaluerefs(self):
        if False:
            i = 10
            return i + 15
        "Return an iterator that yields the weak references to the values.\n\n        The references are not guaranteed to be 'live' at the time\n        they are used, so the result of calling the references needs\n        to be checked before being used.  This can be used to avoid\n        creating references that will cause the garbage collector to\n        keep the values around longer than needed.\n\n        "
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            yield from self.data.values()

    def values(self):
        if False:
            return 10
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            for wr in self.data.values():
                obj = wr()
                if obj is not None:
                    yield obj

    def popitem(self):
        if False:
            while True:
                i = 10
        if self._pending_removals:
            self._commit_removals()
        while True:
            (key, wr) = self.data.popitem()
            o = wr()
            if o is not None:
                return (key, o)

    def pop(self, key, *args):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        try:
            o = self.data.pop(key)()
        except KeyError:
            o = None
        if o is None:
            if args:
                return args[0]
            else:
                raise KeyError(key)
        else:
            return o

    def setdefault(self, key, default=None):
        if False:
            i = 10
            return i + 15
        try:
            o = self.data[key]()
        except KeyError:
            o = None
        if o is None:
            if self._pending_removals:
                self._commit_removals()
            self.data[key] = KeyedRef(default, self._remove, key)
            return default
        else:
            return o

    def update(self, other=None, /, **kwargs):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        d = self.data
        if other is not None:
            if not hasattr(other, 'items'):
                other = dict(other)
            for (key, o) in other.items():
                d[key] = KeyedRef(o, self._remove, key)
        for (key, o) in kwargs.items():
            d[key] = KeyedRef(o, self._remove, key)

    def valuerefs(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a list of weak references to the values.\n\n        The references are not guaranteed to be 'live' at the time\n        they are used, so the result of calling the references needs\n        to be checked before being used.  This can be used to avoid\n        creating references that will cause the garbage collector to\n        keep the values around longer than needed.\n\n        "
        if self._pending_removals:
            self._commit_removals()
        return list(self.data.values())

    def __ior__(self, other):
        if False:
            return 10
        self.update(other)
        return self

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if False:
            return 10
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

class KeyedRef(ref):
    """Specialized reference that includes a key corresponding to the value.

    This is used in the WeakValueDictionary to avoid having to create
    a function object for each key stored in the mapping.  A shared
    callback object can use the 'key' attribute of a KeyedRef instead
    of getting a reference to the key from an enclosing scope.

    """
    __slots__ = ('key',)

    def __new__(type, ob, callback, key):
        if False:
            i = 10
            return i + 15
        self = ref.__new__(type, ob, callback)
        self.key = key
        return self

    def __init__(self, ob, callback, key):
        if False:
            return 10
        super().__init__(ob, callback)

class WeakKeyDictionary(_collections_abc.MutableMapping):
    """ Mapping class that references keys weakly.

    Entries in the dictionary will be discarded when there is no
    longer a strong reference to the key. This can be used to
    associate additional data with an object owned by other parts of
    an application without adding attributes to those objects. This
    can be especially useful with objects that override attribute
    accesses.
    """

    def __init__(self, dict=None):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
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
        del self.data[ref(key)]

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.data[ref(key)]

    def __len__(self):
        if False:
            while True:
                i = 10
        if self._dirty_len and self._pending_removals:
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s at %#x>' % (self.__class__.__name__, id(self))

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self.data[ref(key, self._remove)] = value

    def copy(self):
        if False:
            return 10
        new = WeakKeyDictionary()
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
            i = 10
            return i + 15
        return self.data.get(ref(key), default)

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        try:
            wr = ref(key)
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        with _IterationGuard(self):
            for (wr, value) in self.data.items():
                key = wr()
                if key is not None:
                    yield (key, value)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj
    __iter__ = keys

    def values(self):
        if False:
            print('Hello World!')
        with _IterationGuard(self):
            for (wr, value) in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        self._dirty_len = True
        return self.data.pop(ref(key), *args)

    def setdefault(self, key, default=None):
        if False:
            i = 10
            return i + 15
        return self.data.setdefault(ref(key, self._remove), default)

    def update(self, dict=None, /, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        d = self.data
        if dict is not None:
            if not hasattr(dict, 'items'):
                dict = type({})(dict)
            for (key, value) in dict.items():
                d[ref(key, self._remove)] = value
        if len(kwargs):
            self.update(kwargs)

    def __ior__(self, other):
        if False:
            while True:
                i = 10
        self.update(other)
        return self

    def __or__(self, other):
        if False:
            return 10
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

class finalize:
    """Class for finalization of weakrefable objects

    finalize(obj, func, *args, **kwargs) returns a callable finalizer
    object which will be called when obj is garbage collected. The
    first time the finalizer is called it evaluates func(*arg, **kwargs)
    and returns the result. After this the finalizer is dead, and
    calling it just returns None.

    When the program exits any remaining finalizers for which the
    atexit attribute is true will be run in reverse order of creation.
    By default atexit is true.
    """
    __slots__ = ()
    _registry = {}
    _shutdown = False
    _index_iter = itertools.count()
    _dirty = False
    _registered_with_atexit = False

    class _Info:
        __slots__ = ('weakref', 'func', 'args', 'kwargs', 'atexit', 'index')

    def __init__(self, obj, func, /, *args, **kwargs):
        if False:
            print('Hello World!')
        if not self._registered_with_atexit:
            import atexit
            atexit.register(self._exitfunc)
            finalize._registered_with_atexit = True
        info = self._Info()
        info.weakref = ref(obj, self)
        info.func = func
        info.args = args
        info.kwargs = kwargs or None
        info.atexit = True
        info.index = next(self._index_iter)
        self._registry[self] = info
        finalize._dirty = True

    def __call__(self, _=None):
        if False:
            return 10
        'If alive then mark as dead and return func(*args, **kwargs);\n        otherwise return None'
        info = self._registry.pop(self, None)
        if info and (not self._shutdown):
            return info.func(*info.args, **info.kwargs or {})

    def detach(self):
        if False:
            i = 10
            return i + 15
        'If alive then mark as dead and return (obj, func, args, kwargs);\n        otherwise return None'
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is not None and self._registry.pop(self, None):
            return (obj, info.func, info.args, info.kwargs or {})

    def peek(self):
        if False:
            print('Hello World!')
        'If alive then return (obj, func, args, kwargs);\n        otherwise return None'
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is not None:
            return (obj, info.func, info.args, info.kwargs or {})

    @property
    def alive(self):
        if False:
            while True:
                i = 10
        'Whether finalizer is alive'
        return self in self._registry

    @property
    def atexit(self):
        if False:
            i = 10
            return i + 15
        'Whether finalizer should be called at exit'
        info = self._registry.get(self)
        return bool(info) and info.atexit

    @atexit.setter
    def atexit(self, value):
        if False:
            for i in range(10):
                print('nop')
        info = self._registry.get(self)
        if info:
            info.atexit = bool(value)

    def __repr__(self):
        if False:
            print('Hello World!')
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is None:
            return '<%s object at %#x; dead>' % (type(self).__name__, id(self))
        else:
            return '<%s object at %#x; for %r at %#x>' % (type(self).__name__, id(self), type(obj).__name__, id(obj))

    @classmethod
    def _select_for_exit(cls):
        if False:
            return 10
        L = [(f, i) for (f, i) in cls._registry.items() if i.atexit]
        L.sort(key=lambda item: item[1].index)
        return [f for (f, i) in L]

    @classmethod
    def _exitfunc(cls):
        if False:
            print('Hello World!')
        reenable_gc = False
        try:
            if cls._registry:
                import gc
                if gc.isenabled():
                    reenable_gc = True
                    gc.disable()
                pending = None
                while True:
                    if pending is None or finalize._dirty:
                        pending = cls._select_for_exit()
                        finalize._dirty = False
                    if not pending:
                        break
                    f = pending.pop()
                    try:
                        f()
                    except Exception:
                        sys.excepthook(*sys.exc_info())
                    assert f not in cls._registry
        finally:
            finalize._shutdown = True
            if reenable_gc:
                gc.enable()