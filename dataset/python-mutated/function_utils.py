"""Utility to retrieve function args."""
import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

def _is_bound_method(fn):
    if False:
        return 10
    (_, fn) = tf_decorator.unwrap(fn)
    return tf_inspect.ismethod(fn) and fn.__self__ is not None

def _is_callable_object(obj):
    if False:
        for i in range(10):
            print('nop')
    return hasattr(obj, '__call__') and tf_inspect.ismethod(obj.__call__)

def fn_args(fn):
    if False:
        for i in range(10):
            print('nop')
    'Get argument names for function-like object.\n\n  Args:\n    fn: Function, or function-like object (e.g., result of `functools.partial`).\n\n  Returns:\n    `tuple` of string argument names.\n\n  Raises:\n    ValueError: if partial function has positionally bound arguments\n  '
    if isinstance(fn, functools.partial):
        args = fn_args(fn.func)
        args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
    else:
        if _is_callable_object(fn):
            fn = fn.__call__
        args = tf_inspect.getfullargspec(fn).args
        if _is_bound_method(fn) and args:
            args.pop(0)
    return tuple(args)

def has_kwargs(fn):
    if False:
        return 10
    'Returns whether the passed callable has **kwargs in its signature.\n\n  Args:\n    fn: Function, or function-like object (e.g., result of `functools.partial`).\n\n  Returns:\n    `bool`: if `fn` has **kwargs in its signature.\n\n  Raises:\n     `TypeError`: If fn is not a Function, or function-like object.\n  '
    if isinstance(fn, functools.partial):
        fn = fn.func
    elif _is_callable_object(fn):
        fn = fn.__call__
    elif not callable(fn):
        raise TypeError(f'Argument `fn` should be a callable. Received: fn={fn} (of type {type(fn)})')
    return tf_inspect.getfullargspec(fn).varkw is not None

def get_func_name(func):
    if False:
        while True:
            i = 10
    'Returns name of passed callable.'
    (_, func) = tf_decorator.unwrap(func)
    if callable(func):
        if tf_inspect.isfunction(func):
            return func.__name__
        elif tf_inspect.ismethod(func):
            return '%s.%s' % (six.get_method_self(func).__class__.__name__, six.get_method_function(func).__name__)
        else:
            return str(type(func))
    else:
        raise ValueError(f'Argument `func` must be a callable. Received func={func} (of type {type(func)})')

def get_func_code(func):
    if False:
        for i in range(10):
            print('nop')
    'Returns func_code of passed callable, or None if not available.'
    (_, func) = tf_decorator.unwrap(func)
    if callable(func):
        if tf_inspect.isfunction(func) or tf_inspect.ismethod(func):
            return six.get_function_code(func)
        try:
            return six.get_function_code(func.__call__)
        except AttributeError:
            return None
    else:
        raise ValueError(f'Argument `func` must be a callable. Received func={func} (of type {type(func)})')
_rewriter_config_optimizer_disabled = None

def get_disabled_rewriter_config():
    if False:
        while True:
            i = 10
    global _rewriter_config_optimizer_disabled
    if _rewriter_config_optimizer_disabled is None:
        config = config_pb2.ConfigProto()
        rewriter_config = config.graph_options.rewrite_options
        rewriter_config.disable_meta_optimizer = True
        _rewriter_config_optimizer_disabled = config.SerializeToString()
    return _rewriter_config_optimizer_disabled