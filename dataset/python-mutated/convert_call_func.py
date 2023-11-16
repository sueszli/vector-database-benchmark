import collections
import copy
import functools
import inspect
import logging
import os
import pdb
import re
from typing import Any, List
import numpy
from paddle.nn import Layer
from .convert_operators import convert_enumerate, convert_len, convert_print, convert_range, convert_zip
from .logging_utils import TranslatorLogger
from .program_translator import CONVERSION_OPTIONS, StaticFunction, convert_to_static, unwrap_decorators
from .utils import is_builtin, is_paddle_func, unwrap
__all__ = []
translator_logger = TranslatorLogger()

class ConversionOptions:
    """
    A container for conversion flags of a function in dynamic-to-static.

    Attributes:
        not_convert(bool): An attribute indicates that the function won't be converted in dynamic-to-static.

    NOTE(liym27): More attributes and methods can be added in this class.
    """

    def __init__(self, not_convert=False):
        if False:
            print('Hello World!')
        self.not_convert = not_convert

    def attach(self, func):
        if False:
            while True:
                i = 10
        if inspect.ismethod(func):
            func = func.__func__
        if inspect.isfunction(func):
            setattr(func, CONVERSION_OPTIONS, self)
        else:
            translator_logger.warn('Only support @not_to_static to type(function) or type(method), but recevied {}'.format(type(func)))

def builtin_modules():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return builtin modules.\n    '
    modules = [copy, collections, inspect, logging, numpy, os, pdb, re]
    try:
        import six
        modules.append(six)
    except ImportError:
        pass
    return modules
BUILTIN_LIKELY_MODULES = builtin_modules()

def add_ignore_module(modules: List[Any]):
    if False:
        return 10
    '\n    Adds modules that ignore transcription\n    '
    global BUILTIN_LIKELY_MODULES
    for module in modules:
        if module not in BUILTIN_LIKELY_MODULES:
            BUILTIN_LIKELY_MODULES.append(module)

def is_unsupported(func):
    if False:
        return 10
    '\n    Checks whether the func is supported by dygraph to static graph.\n    '
    for m in BUILTIN_LIKELY_MODULES:
        for v in m.__dict__.values():
            if not callable(v):
                continue
            if func is v:
                translator_logger.log(2, f'Whitelist: {func} is part of built-in module and does not have to be transformed.')
                return True
    from paddle.nn import Sequential
    PADDLE_NEED_CONVERT_APIS = [Sequential]
    if type(func) in PADDLE_NEED_CONVERT_APIS:
        return False
    if is_paddle_func(func):
        translator_logger.log(2, f'Whitelist: {func} is part of Paddle module and does not have to be transformed.')
        return True

def convert_call(func):
    if False:
        print('Hello World!')
    "\n    Converts a function call which needs to be transformed to static function.\n\n    Args:\n        func (callable): A callable function or method to convert.\n\n    Returns:\n        Callable: A converted function.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')\n            >>> import paddle\n            >>> from paddle.jit.dy2static import Call\n\n            >>> paddle.enable_static()\n            >>> def dyfunc(x):\n            ...     if paddle.mean(x) < 0:\n            ...         x_v = x - 1\n            ...     else:\n            ...         x_v = x + 1\n            ...     return x_v\n            ...\n            >>> new_func = Call(dyfunc)\n            >>> x = paddle.tensor.manipulation.fill_constant(shape=[3, 3], value=0, dtype='float64')\n            >>> x_v = new_func(x)\n\n            >>> exe = paddle.static.Executor(paddle.CPUPlace())\n            >>> out = exe.run(fetch_list=[x_v])\n            >>> print(out[0])\n            [[1. 1. 1.]\n             [1. 1. 1.]\n             [1. 1. 1.]]\n\n    "
    translator_logger.log(1, f'Convert callable object: convert {func}.')
    func_self = None
    converted_call = None
    (_, func) = unwrap_decorators(func)
    options = getattr(func, CONVERSION_OPTIONS, None)
    if options is not None and options.not_convert:
        translator_logger.log(2, f"{func} is not converted when it is decorated by 'paddle.jit.not_to_static'.")
        return func
    if is_builtin(func, 'len'):
        return convert_len
    if is_builtin(func, 'zip'):
        return convert_zip
    if is_builtin(func, 'range'):
        return convert_range
    if is_builtin(func, 'enumerate'):
        return convert_enumerate
    if is_builtin(func, 'print'):
        return convert_print
    if is_builtin(func) or is_unsupported(func):
        return func
    if inspect.isgeneratorfunction(func):
        number_of_stars = 30
        translator_logger.warn('\n\n' + '*' * number_of_stars + "\nYour function:`{}` doesn't support to transform to static function because it is a generator function, it will be run as-is.".format(func.__name__) + '\n' + '*' * number_of_stars + '\n\n')
        return func
    if inspect.isfunction(func):
        if func.__name__ == '<lambda>':
            return func
        try:
            _origfunc = unwrap(func)
            global_functions = set()
            for fn in _origfunc.__globals__.values():
                if inspect.isfunction(fn):
                    global_functions.add(fn)
                elif isinstance(fn, StaticFunction):
                    (_, fn) = unwrap_decorators(fn)
                    global_functions.add(fn)
                elif inspect.isclass(fn):
                    if isinstance(fn.__dict__.get(func.__name__, None), staticmethod):
                        global_functions.add(func)
            if func in global_functions:
                converted_call = convert_to_static(func)
                func_self = getattr(func, '__self__', None)
            else:
                translator_logger.warn(f"{func} doesn't have to be transformed to static function because it has been transformed before, it will be run as-is.")
                converted_call = func
        except AttributeError:
            converted_call = None
        except OSError:
            converted_call = None
    elif inspect.ismethod(func):
        try:
            converted_call = convert_to_static(func)
            func_self = getattr(func, '__self__', None)
        except OSError:
            converted_call = None
    elif hasattr(func, '__class__') and callable(func.__class__):
        if hasattr(func, 'forward') and isinstance(func, Layer):
            try:
                (_, forward_func) = unwrap_decorators(func.forward)
                func._original_funcs['forward'] = forward_func.__func__
                forward_func = convert_to_static(forward_func)
                func.forward = forward_func.__get__(func)
            except (OSError, TypeError):
                func_self = None if func_self else func_self
            converted_call = func
        else:
            try:
                call_func = func.__class__.__call__
                converted_call = convert_to_static(call_func)
                func_self = func
            except (OSError, TypeError):
                func_self = None if func_self else func_self
    else:
        raise NotImplementedError(f'Callable {func} can not be transformed at present.')
    if converted_call is None:
        translator_logger.warn(f"{func} doesn't have to be transformed to static function, and it will be run as-is.")
        return func
    if func_self is not None:
        converted_call = functools.partial(converted_call, func_self)
    return converted_call