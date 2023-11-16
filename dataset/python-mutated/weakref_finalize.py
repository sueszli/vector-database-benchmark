"""
backports.weakref_finalize
~~~~~~~~~~~~~~~~~~

Backports the Python 3 ``weakref.finalize`` method.
"""
from __future__ import absolute_import
import itertools
import sys
from weakref import ref
__all__ = ['weakref_finalize']

class weakref_finalize(object):
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

    class _Info(object):
        __slots__ = ('weakref', 'func', 'args', 'kwargs', 'atexit', 'index')

    def __init__(self, obj, func, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self._registered_with_atexit:
            import atexit
            atexit.register(self._exitfunc)
            weakref_finalize._registered_with_atexit = True
        info = self._Info()
        info.weakref = ref(obj, self)
        info.func = func
        info.args = args
        info.kwargs = kwargs or None
        info.atexit = True
        info.index = next(self._index_iter)
        self._registry[self] = info
        weakref_finalize._dirty = True

    def __call__(self, _=None):
        if False:
            print('Hello World!')
        'If alive then mark as dead and return func(*args, **kwargs);\n        otherwise return None'
        info = self._registry.pop(self, None)
        if info and (not self._shutdown):
            return info.func(*info.args, **info.kwargs or {})

    def detach(self):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        'Whether finalizer is alive'
        return self in self._registry

    @property
    def atexit(self):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is None:
            return '<%s object at %#x; dead>' % (type(self).__name__, id(self))
        else:
            return '<%s object at %#x; for %r at %#x>' % (type(self).__name__, id(self), type(obj).__name__, id(obj))

    @classmethod
    def _select_for_exit(cls):
        if False:
            for i in range(10):
                print('nop')
        L = [(f, i) for (f, i) in cls._registry.items() if i.atexit]
        L.sort(key=lambda item: item[1].index)
        return [f for (f, i) in L]

    @classmethod
    def _exitfunc(cls):
        if False:
            while True:
                i = 10
        reenable_gc = False
        try:
            if cls._registry:
                import gc
                if gc.isenabled():
                    reenable_gc = True
                    gc.disable()
                pending = None
                while True:
                    if pending is None or weakref_finalize._dirty:
                        pending = cls._select_for_exit()
                        weakref_finalize._dirty = False
                    if not pending:
                        break
                    f = pending.pop()
                    try:
                        f()
                    except Exception:
                        sys.excepthook(*sys.exc_info())
                    assert f not in cls._registry
        finally:
            weakref_finalize._shutdown = True
            if reenable_gc:
                gc.enable()