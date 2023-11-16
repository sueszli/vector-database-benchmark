"""
Cooperative ``contextvars`` module.

This module was added to Python 3.7. The gevent version is available
on all supported versions of Python. However, see an important note
about gevent 20.9.

Context variables are like greenlet-local variables, just more
inconvenient to use. They were designed to work around limitations in
:mod:`asyncio` and are rarely needed by greenlet-based code.

The primary difference is that snapshots of the state of all context
variables in a given greenlet can be taken, and later restored for
execution; modifications to context variables are "scoped" to the
duration that a particular context is active. (This state-restoration
support is rarely useful for greenlets because instead of always
running "tasks" sequentially within a single thread like `asyncio`
does, greenlet-based code usually spawns new greenlets to handle each
task.)

The gevent implementation is based on the Python reference implementation
from :pep:`567` and doesn't have much optimization. In particular, setting
context values isn't constant time.

.. versionadded:: 1.5a3
.. versionchanged:: 20.9.0
   On Python 3.7 and above, this module is no longer monkey-patched
   in place of the standard library version.
   gevent depends on greenlet 0.4.17 which includes support for context variables.
   This means that any number of greenlets can be running any number of asyncio tasks
   each with their own context variables. This module is only greenlet aware, not
   asyncio task aware, so its use is not recommended on Python 3.7 and above.

   On previous versions of Python, this module continues to be a solution for
   backporting code. It is also available if you wish to use the contextvar API
   in a strictly greenlet-local manner.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__all__ = ['ContextVar', 'Context', 'copy_context', 'Token']
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from gevent._util import _NONE
from gevent.local import local
__stdlib_expected__ = __all__
__implements__ = __stdlib_expected__

class _ContextState(local):

    def __init__(self):
        if False:
            print('Hello World!')
        self.context = Context()

def _not_base_type(cls):
    if False:
        for i in range(10):
            print('nop')
    raise TypeError('not an acceptable base type')

class _ContextData(object):
    """
    A copy-on-write immutable mapping from ContextVar
    keys to arbitrary values. Setting values requires a
    copy, making it O(n), not O(1).
    """
    __slots__ = ('_mapping',)

    def __init__(self):
        if False:
            while True:
                i = 10
        self._mapping = {}

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self._mapping[key]

    def __contains__(self, key):
        if False:
            return 10
        return key in self._mapping

    def __len__(self):
        if False:
            return 10
        return len(self._mapping)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._mapping)

    def set(self, key, value):
        if False:
            i = 10
            return i + 15
        copy = _ContextData()
        copy._mapping = self._mapping.copy()
        copy._mapping[key] = value
        return copy

    def delete(self, key):
        if False:
            i = 10
            return i + 15
        copy = _ContextData()
        copy._mapping = self._mapping.copy()
        del copy._mapping[key]
        return copy

class ContextVar(object):
    """
    Implementation of :class:`contextvars.ContextVar`.
    """
    __slots__ = ('_name', '_default')

    def __init__(self, name, default=_NONE):
        if False:
            return 10
        self._name = name
        self._default = default
    __init_subclass__ = classmethod(_not_base_type)

    @classmethod
    def __class_getitem__(cls, _):
        if False:
            print('Hello World!')
        return cls

    @property
    def name(self):
        if False:
            return 10
        return self._name

    def get(self, default=_NONE):
        if False:
            print('Hello World!')
        context = _context_state.context
        try:
            return context[self]
        except KeyError:
            pass
        if default is not _NONE:
            return default
        if self._default is not _NONE:
            return self._default
        raise LookupError

    def set(self, value):
        if False:
            return 10
        context = _context_state.context
        return context._set_value(self, value)

    def reset(self, token):
        if False:
            i = 10
            return i + 15
        token._reset(self)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s.%s name=%r default=%r at 0x%x>' % (type(self).__module__, type(self).__name__, self._name, self._default, id(self))

class Token(object):
    """
    Opaque implementation of :class:`contextvars.Token`.
    """
    MISSING = _NONE
    __slots__ = ('_context', '_var', '_old_value', '_used')

    def __init__(self, context, var, old_value):
        if False:
            return 10
        self._context = context
        self._var = var
        self._old_value = old_value
        self._used = False
    __init_subclass__ = classmethod(_not_base_type)

    @property
    def var(self):
        if False:
            i = 10
            return i + 15
        '\n        A read-only attribute pointing to the variable that created the token\n        '
        return self._var

    @property
    def old_value(self):
        if False:
            i = 10
            return i + 15
        "\n        A read-only attribute set to the value the variable had before\n        the ``set()`` call, or to :attr:`MISSING` if the variable wasn't set\n        before.\n        "
        return self._old_value

    def _reset(self, var):
        if False:
            while True:
                i = 10
        if self._used:
            raise RuntimeError('Taken has already been used once')
        if self._var is not var:
            raise ValueError('Token was created by a different ContextVar')
        if self._context is not _context_state.context:
            raise ValueError('Token was created in a different Context')
        self._used = True
        if self._old_value is self.MISSING:
            self._context._delete(var)
        else:
            self._context._reset_value(var, self._old_value)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s.%s%s var=%r at 0x%x>' % (type(self).__module__, type(self).__name__, ' used' if self._used else '', self._var, id(self))

class Context(Mapping):
    """
    Implementation of :class:`contextvars.Context`
    """
    __slots__ = ('_data', '_prev_context')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates an empty context.\n        '
        self._data = _ContextData()
        self._prev_context = None
    __init_subclass__ = classmethod(_not_base_type)

    def run(self, function, *args, **kwargs):
        if False:
            return 10
        if self._prev_context is not None:
            raise RuntimeError('Cannot enter context; %s is already entered' % (self,))
        self._prev_context = _context_state.context
        try:
            _context_state.context = self
            return function(*args, **kwargs)
        finally:
            _context_state.context = self._prev_context
            self._prev_context = None

    def copy(self):
        if False:
            while True:
                i = 10
        '\n        Return a shallow copy.\n        '
        result = Context()
        result._data = self._data
        return result

    def _set_value(self, var, value):
        if False:
            i = 10
            return i + 15
        try:
            old_value = self._data[var]
        except KeyError:
            old_value = Token.MISSING
        self._data = self._data.set(var, value)
        return Token(self, var, old_value)

    def _delete(self, var):
        if False:
            while True:
                i = 10
        self._data = self._data.delete(var)

    def _reset_value(self, var, old_value):
        if False:
            for i in range(10):
                print('nop')
        self._data = self._data.set(var, old_value)

    @staticmethod
    def __check_key(key):
        if False:
            print('Hello World!')
        if type(key) is not ContextVar:
            raise TypeError('ContextVar key was expected')

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        self.__check_key(key)
        return self._data[key]

    def __contains__(self, key):
        if False:
            return 10
        self.__check_key(key)
        return key in self._data

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._data)

    def __iter__(self):
        if False:
            return 10
        return iter(self._data)

def copy_context():
    if False:
        return 10
    '\n    Return a shallow copy of the current context.\n    '
    return _context_state.context.copy()
_context_state = _ContextState()