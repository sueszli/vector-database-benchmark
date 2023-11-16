import functools
import inspect
import reprlib
import sys
import traceback
from . import constants

def _get_function_source(func):
    if False:
        while True:
            i = 10
    func = inspect.unwrap(func)
    if inspect.isfunction(func):
        code = func.__code__
        return (code.co_filename, code.co_firstlineno)
    if isinstance(func, functools.partial):
        return _get_function_source(func.func)
    if isinstance(func, functools.partialmethod):
        return _get_function_source(func.func)
    return None

def _format_callback_source(func, args):
    if False:
        print('Hello World!')
    func_repr = _format_callback(func, args, None)
    source = _get_function_source(func)
    if source:
        func_repr += f' at {source[0]}:{source[1]}'
    return func_repr

def _format_args_and_kwargs(args, kwargs):
    if False:
        i = 10
        return i + 15
    "Format function arguments and keyword arguments.\n\n    Special case for a single parameter: ('hello',) is formatted as ('hello').\n    "
    items = []
    if args:
        items.extend((reprlib.repr(arg) for arg in args))
    if kwargs:
        items.extend((f'{k}={reprlib.repr(v)}' for (k, v) in kwargs.items()))
    return '({})'.format(', '.join(items))

def _format_callback(func, args, kwargs, suffix=''):
    if False:
        return 10
    if isinstance(func, functools.partial):
        suffix = _format_args_and_kwargs(args, kwargs) + suffix
        return _format_callback(func.func, func.args, func.keywords, suffix)
    if hasattr(func, '__qualname__') and func.__qualname__:
        func_repr = func.__qualname__
    elif hasattr(func, '__name__') and func.__name__:
        func_repr = func.__name__
    else:
        func_repr = repr(func)
    func_repr += _format_args_and_kwargs(args, kwargs)
    if suffix:
        func_repr += suffix
    return func_repr

def extract_stack(f=None, limit=None):
    if False:
        return 10
    'Replacement for traceback.extract_stack() that only does the\n    necessary work for asyncio debug mode.\n    '
    if f is None:
        f = sys._getframe().f_back
    if limit is None:
        limit = constants.DEBUG_STACK_DEPTH
    stack = traceback.StackSummary.extract(traceback.walk_stack(f), limit=limit, lookup_lines=False)
    stack.reverse()
    return stack