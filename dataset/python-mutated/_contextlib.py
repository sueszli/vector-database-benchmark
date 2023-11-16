import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)

def _wrap_generator(ctx_factory, func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap each generator invocation with the context manager factory.\n\n    The input should be a function that returns a context manager,\n    not a context manager itself, to handle one-shot context managers.\n    '

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        gen = func(*args, **kwargs)
        try:
            with ctx_factory():
                response = gen.send(None)
            while True:
                try:
                    request = (yield response)
                except GeneratorExit:
                    with ctx_factory():
                        gen.close()
                    raise
                except BaseException:
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())
                else:
                    with ctx_factory():
                        response = gen.send(request)
        except StopIteration as e:
            return e.value
    return generator_context

def context_decorator(ctx, func):
    if False:
        i = 10
        return i + 15
    '\n    Like contextlib.ContextDecorator.\n\n    But with the following differences:\n    1. Is done by wrapping, rather than inheritance, so it works with context\n       managers that are implemented from C and thus cannot easily inherit from\n       Python classes\n    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)\n    3. Errors out if you try to wrap a class, because it is ambiguous whether\n       or not you intended to wrap only the constructor\n\n    The input argument can either be a context manager (in which case it must\n    be a multi-shot context manager that can be directly invoked multiple times)\n    or a callable that produces a context manager.\n    '
    assert not (callable(ctx) and hasattr(ctx, '__enter__')), f'Passed in {ctx} is both callable and also a valid context manager (has __enter__), making it ambiguous which interface to use.  If you intended to pass a context manager factory, rewrite your call as context_decorator(lambda: ctx()); if you intended to pass a context manager directly, rewrite your call as context_decorator(lambda: ctx)'
    if not callable(ctx):

        def ctx_factory():
            if False:
                print('Hello World!')
            return ctx
    else:
        ctx_factory = ctx
    if inspect.isclass(func):
        raise RuntimeError('Cannot decorate classes; it is ambiguous whether or not only the constructor or all methods should have the context manager applied; additionally, decorating a class at definition-site will prevent use of the identifier as a conventional type.  To specify which methods to decorate, decorate each of them individually.')
    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        if False:
            return 10
        with ctx_factory():
            return func(*args, **kwargs)
    return decorate_context

class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        if False:
            while True:
                i = 10
        if inspect.isclass(orig_func):
            warnings.warn('Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.')
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func
        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if False:
            return 10
        raise NotImplementedError

    def clone(self):
        if False:
            while True:
                i = 10
        return self.__class__()

class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""

    def __new__(cls, orig_func=None):
        if False:
            for i in range(10):
                print('nop')
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)