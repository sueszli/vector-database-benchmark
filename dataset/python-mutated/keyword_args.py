"""Keyword args functions."""
import functools
from tensorflow.python.util import decorator_utils

def keyword_args_only(func):
    if False:
        for i in range(10):
            print('nop')
    'Decorator for marking specific function accepting keyword args only.\n\n  This decorator raises a `ValueError` if the input `func` is called with any\n  non-keyword args. This prevents the caller from providing the arguments in\n  wrong order.\n\n  Args:\n    func: The function or method needed to be decorated.\n\n  Returns:\n    Decorated function or method.\n\n  Raises:\n    ValueError: If `func` is not callable.\n  '
    decorator_utils.validate_callable(func, 'keyword_args_only')

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Keyword args only wrapper.'
        if args:
            raise ValueError(f'The function {func.__name__} only accepts keyword arguments. Do not pass positional arguments. Received the following positional arguments: {args}')
        return func(**kwargs)
    return new_func