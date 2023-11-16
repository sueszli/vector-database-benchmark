"""
Module for testing automatic garbage collection of objects

.. autosummary::
   :toctree: generated/

   set_gc_state - enable or disable garbage collection
   gc_state - context manager for given state of garbage collector
   assert_deallocated - context manager to check for circular references on object

"""
import weakref
import gc
from contextlib import contextmanager
from platform import python_implementation
__all__ = ['set_gc_state', 'gc_state', 'assert_deallocated']
IS_PYPY = python_implementation() == 'PyPy'

class ReferenceError(AssertionError):
    pass

def set_gc_state(state):
    if False:
        return 10
    ' Set status of garbage collector '
    if gc.isenabled() == state:
        return
    if state:
        gc.enable()
    else:
        gc.disable()

@contextmanager
def gc_state(state):
    if False:
        while True:
            i = 10
    ' Context manager to set state of garbage collector to `state`\n\n    Parameters\n    ----------\n    state : bool\n        True for gc enabled, False for disabled\n\n    Examples\n    --------\n    >>> with gc_state(False):\n    ...     assert not gc.isenabled()\n    >>> with gc_state(True):\n    ...     assert gc.isenabled()\n    '
    orig_state = gc.isenabled()
    set_gc_state(state)
    yield
    set_gc_state(orig_state)

@contextmanager
def assert_deallocated(func, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Context manager to check that object is deallocated\n\n    This is useful for checking that an object can be freed directly by\n    reference counting, without requiring gc to break reference cycles.\n    GC is disabled inside the context manager.\n\n    This check is not available on PyPy.\n\n    Parameters\n    ----------\n    func : callable\n        Callable to create object to check\n    \\*args : sequence\n        positional arguments to `func` in order to create object to check\n    \\*\\*kwargs : dict\n        keyword arguments to `func` in order to create object to check\n\n    Examples\n    --------\n    >>> class C: pass\n    >>> with assert_deallocated(C) as c:\n    ...     # do something\n    ...     del c\n\n    >>> class C:\n    ...     def __init__(self):\n    ...         self._circular = self # Make circular reference\n    >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL\n    ...     # do something\n    ...     del c\n    Traceback (most recent call last):\n        ...\n    ReferenceError: Remaining reference(s) to object\n    '
    if IS_PYPY:
        raise RuntimeError('assert_deallocated is unavailable on PyPy')
    with gc_state(False):
        obj = func(*args, **kwargs)
        ref = weakref.ref(obj)
        yield obj
        del obj
        if ref() is not None:
            raise ReferenceError('Remaining reference(s) to object')