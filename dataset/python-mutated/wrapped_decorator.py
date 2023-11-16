import contextlib
import decorator
__all__ = []

def wrap_decorator(decorator_func):
    if False:
        i = 10
        return i + 15

    @decorator.decorator
    def __impl__(func, *args, **kwargs):
        if False:
            print('Hello World!')
        wrapped_func = decorator_func(func)
        return wrapped_func(*args, **kwargs)
    return __impl__
signature_safe_contextmanager = wrap_decorator(contextlib.contextmanager)