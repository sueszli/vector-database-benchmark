"""Tools useful for creating decorators, and other high-level callables."""
import functools
import inspect
import types
from typing import Type, Callable
_MAGIC_STATICMETHODS = {'__new__'}
_MAGIC_CLASSMETHODS = {'__init_subclass__', '__prepare__'}

class _lift_to_method:
    """A decorator that ensures that an input callable object implements ``__get__``.  It is
    returned unchanged if so, otherwise it is turned into the default implementation for functions,
    which makes them bindable to instances.

    Python-space functions and lambdas already have this behaviour, but builtins like ``print``
    don't; using this class allows us to do::

        wrap_method(MyClass, "maybe_mutates_arguments", before=print, after=print)

    to simply print all the arguments on entry and exit of the function, which otherwise wouldn't be
    valid, since ``print`` isn't a descriptor.
    """
    __slots__ = ('_method',)

    def __new__(cls, method):
        if False:
            i = 10
            return i + 15
        if hasattr(method, '__get__'):
            return method
        return super().__new__(cls)

    def __init__(self, method):
        if False:
            print('Hello World!')
        if method is self:
            return
        self._method = method

    def __get__(self, obj, objtype):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            return self._method
        return types.MethodType(self._method, obj)

class _WrappedMethod:
    """Descriptor which calls its two arguments in succession, correctly handling instance- and
    class-method calls.

    It is intended that this class will replace the attribute that ``inner`` previously was on a
    class or instance.  When accessed as that attribute, this descriptor will behave it is the same
    function call, but with the ``function`` called before or after.
    """
    __slots__ = ('_method_decorator', '_method_has_get', '_method', '_before', '_after')

    def __init__(self, method, before=None, after=None):
        if False:
            while True:
                i = 10
        if isinstance(method, (classmethod, staticmethod)):
            self._method_decorator = type(method)
        elif isinstance(method, type(self)):
            self._method_decorator = method._method_decorator
        elif getattr(method, '__name__', None) in _MAGIC_STATICMETHODS:
            self._method_decorator = staticmethod
        elif getattr(method, '__name__', None) in _MAGIC_CLASSMETHODS:
            self._method_decorator = classmethod
        else:
            self._method_decorator = _lift_to_method
        before = (self._method_decorator(before),) if before is not None else ()
        after = (self._method_decorator(after),) if after is not None else ()
        if isinstance(method, type(self)):
            self._method = method._method
            self._before = before + method._before
            self._after = method._after + after
        else:
            self._before = before
            self._after = after
            self._method = method
        self._method_has_get = hasattr(self._method, '__get__')

    def __get__(self, obj, objtype=None):
        if False:
            i = 10
            return i + 15
        method = self._method.__get__(obj, objtype) if self._method_has_get else self._method

        @functools.wraps(method)
        def out(*args, **kwargs):
            if False:
                while True:
                    i = 10
            for callback in self._before:
                callback.__get__(obj, objtype)(*args, **kwargs)
            retval = method(*args, **kwargs)
            for callback in self._after:
                callback.__get__(obj, objtype)(*args, **kwargs)
            return retval
        return out

def wrap_method(cls: Type, name: str, *, before: Callable=None, after: Callable=None):
    if False:
        return 10
    'Wrap the functionality the instance- or class method ``cls.name`` with additional behaviour\n    ``before`` and ``after``.\n\n    This mutates ``cls``, replacing the attribute ``name`` with the new functionality.  This is\n    useful when creating class decorators.  The method is allowed to be defined on any parent class\n    instead.\n\n    If either ``before`` or ``after`` are given, they should be callables with a compatible\n    signature to the method referred to.  They will be called immediately before or after the method\n    as appropriate, and any return value will be ignored.\n\n    Args:\n        cls: the class to modify.\n        name: the name of the method on the class to wrap.\n        before: a callable that should be called before the method that is being wrapped.\n        after: a callable that should be called after the method that is being wrapped.\n\n    Raises:\n        ValueError: if the named method is not defined on the class or any parent class.\n    '
    method = inspect.getattr_static(cls, name)
    setattr(cls, name, _WrappedMethod(method, before, after))