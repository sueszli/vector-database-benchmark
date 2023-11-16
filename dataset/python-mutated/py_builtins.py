"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""
import inspect
from nvidia.dali._autograph.utils import hooks
UNSPECIFIED = object()

def overload_of(f):
    if False:
        for i in range(10):
            print('nop')
    if f in SUPPORTED_BUILTINS:
        return BUILTIN_FUNCTIONS_MAP[f.__name__]
    return f

def _find_originating_frame(caller_fn_scope, innermost=True):
    if False:
        return 10
    'Locates the frame in which `caller_fn_scope` was defined.'
    ctx_frame = inspect.currentframe()
    result = None
    while ctx_frame is not None:
        if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
            result = ctx_frame
            if innermost:
                break
        ctx_frame = ctx_frame.f_back
    assert result is not None, 'the conversion process should ensure the caller_fn_scope is always found somewhere on the call stack'
    return result

def locals_in_original_context(caller_fn_scope):
    if False:
        while True:
            i = 10
    'Executes the locals function in the context of a specified function.'
    return _find_originating_frame(caller_fn_scope, innermost=True).f_locals

def globals_in_original_context(caller_fn_scope):
    if False:
        print('Hello World!')
    'Executes the locals function in the context of a specified function.'
    return _find_originating_frame(caller_fn_scope, innermost=True).f_globals

def eval_in_original_context(f, args, caller_fn_scope):
    if False:
        print('Hello World!')
    'Executes the eval function in the context of a specified function.'
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)
    args = (args[0], ctx_frame.f_globals if len(args) < 2 else args[1], ctx_frame.f_locals if len(args) < 3 else args[2])
    return f(*args)

def super_in_original_context(f, args, caller_fn_scope):
    if False:
        return 10
    'Executes the super function in the context of a specified function.\n\n  See https://docs.python.org/3/library/functions.html#super for the exact\n  details\n\n  Args:\n    f: Callable, typically the super builtin\n    args: List[Any], the original call arguments\n    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function\n      scope of the converted function in which this call was originally made\n\n  Returns:\n    The result of calling `f` as if it was called in the frame indicated by\n      `caller_fn_scope`.\n  '
    if args:
        return f(*args)
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)
    type_arg = ctx_frame.f_locals['__class__']
    self_arg_name = ctx_frame.f_code.co_varnames[0]
    self_arg = ctx_frame.f_locals[self_arg_name]
    return f(type_arg, self_arg)

def abs_(x):
    if False:
        print('Hello World!')
    if hooks._DISPATCH.detect_overload_abs_(x):
        return hooks._DISPATCH.abs_(x)
    return _py_abs(x)

def _py_abs(x):
    if False:
        i = 10
        return i + 15
    return abs(x)

def float_(x=0):
    if False:
        while True:
            i = 10
    if hooks._DISPATCH.detect_overload_float_(x):
        return hooks._DISPATCH.float_(x)
    return _py_float(x)

def _py_float(x):
    if False:
        i = 10
        return i + 15
    return float(x)

def int_(x=0, base=UNSPECIFIED):
    if False:
        i = 10
        return i + 15
    if hooks._DISPATCH.detect_overload_int_(x):
        return hooks._DISPATCH.int_(x, base)
    return _py_int(x, base)

def _py_int(x, base):
    if False:
        i = 10
        return i + 15
    if base is UNSPECIFIED:
        return int(x)
    return int(x, base)

def len_(s):
    if False:
        i = 10
        return i + 15
    if hooks._DISPATCH.detect_overload_len_(s):
        return hooks._DISPATCH.len_(s)
    return _py_len(s)

def _py_len(s):
    if False:
        print('Hello World!')
    return len(s)

def print_(*objects, **kwargs):
    if False:
        return 10
    'Overload of the print builtin.'
    unknown_kwargs = tuple(set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
    if unknown_kwargs:
        raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))
    if hooks._DISPATCH.detect_overload_print_(objects):
        return hooks._DISPATCH.print_(objects, kwargs)
    else:
        _py_print(*objects, **kwargs)

def _py_print(*objects, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    print(*objects, **kwargs)

def min_(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    if hooks._DISPATCH.detect_overload_min_(args):
        return hooks._DISPATCH.min_(*args, **kwargs)
    return _py_min(*args, **kwargs)

def _py_min(*args, **kwargs):
    if False:
        print('Hello World!')
    return min(*args, **kwargs)

def max_(*args, **kwargs):
    if False:
        while True:
            i = 10
    if hooks._DISPATCH.detect_overload_max_(args):
        return hooks._DISPATCH.max_(*args, **kwargs)
    return _py_max(*args, **kwargs)

def _py_max(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return max(*args, **kwargs)

def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
    if False:
        for i in range(10):
            print('nop')
    if hooks._DISPATCH.detect_overload_range_(start_or_stop, stop, step):
        return hooks._DISPATCH.range_(start_or_stop, stop, step)
    return _py_range(start_or_stop, stop, step)

def _py_range(start_or_stop, stop, step):
    if False:
        return 10
    if step is not UNSPECIFIED:
        return range(start_or_stop, stop, step)
    if stop is not UNSPECIFIED:
        return range(start_or_stop, stop)
    return range(start_or_stop)

def enumerate_(s, start=0):
    if False:
        return 10
    if hooks._DISPATCH.detect_overload_enumerate_(s):
        return hooks._DISPATCH.enumerate_(s, start)
    return _py_enumerate(s, start)

def _py_enumerate(s, start=0):
    if False:
        return 10
    return enumerate(s, start)

def zip_(*iterables):
    if False:
        for i in range(10):
            print('nop')
    if hooks._DISPATCH.detect_overload_zip_(iterables):
        return hooks._DISPATCH.zip_(*iterables)
    return _py_zip(*iterables)

def _py_zip(*iterables):
    if False:
        return 10
    return zip(*iterables)

def map_(fn, *iterables):
    if False:
        i = 10
        return i + 15
    if hooks._DISPATCH.detect_overload_map_(iterables):
        return hooks._DISPATCH.map_(fn, *iterables)
    return _py_map(fn, *iterables)

def _py_map(fn, *iterables):
    if False:
        for i in range(10):
            print('nop')
    return map(fn, *iterables)

def next_(iterator, default=UNSPECIFIED):
    if False:
        i = 10
        return i + 15
    if hooks._DISPATCH.detect_overload_next_(iterator):
        return hooks._DISPATCH.next_(iterator, default)
    return next_py(iterator, default)

def next_py(iterator, default=UNSPECIFIED):
    if False:
        print('Hello World!')
    if default is UNSPECIFIED:
        return next(iterator)
    return next(iterator, default)

def filter_(function, iterable):
    if False:
        i = 10
        return i + 15
    if hooks._DISPATCH.detect_overload_filter_(iterable):
        return hooks._DISPATCH.filter_(function, iterable)
    return _py_filter(function, iterable)

def _py_filter(function, iterable):
    if False:
        for i in range(10):
            print('nop')
    return filter(function, iterable)

def any_(iterable):
    if False:
        for i in range(10):
            print('nop')
    if hooks._DISPATCH.detect_overload_any_(iterable):
        return hooks._DISPATCH.any_(iterable)
    return _py_any(iterable)

def _py_any(iterable):
    if False:
        return 10
    return any(iterable)

def all_(iterable):
    if False:
        for i in range(10):
            print('nop')
    if hooks._DISPATCH.detect_overload_all_(iterable):
        return hooks._DISPATCH.all_(iterable)
    return _py_all(iterable)

def _py_all(iterable):
    if False:
        while True:
            i = 10
    return all(iterable)

def sorted_(iterable, key=UNSPECIFIED, reverse=UNSPECIFIED):
    if False:
        while True:
            i = 10
    if hooks._DISPATCH.detect_overload_sorted_(iterable):
        return hooks._DISPATCH.sorted_(iterable, key, reverse)
    return _py_sorted(iterable, key, reverse)

def _py_sorted(iterable, key, reverse):
    if False:
        for i in range(10):
            print('nop')
    if key is not UNSPECIFIED and reverse is UNSPECIFIED:
        return sorted(iterable, key=key)
    if key is UNSPECIFIED and reverse is not UNSPECIFIED:
        return sorted(iterable, reverse=reverse)
    if key is not UNSPECIFIED and reverse is not UNSPECIFIED:
        return sorted(iterable, key=key, reverse=reverse)
    return sorted(iterable)
SUPPORTED_BUILTINS = (abs, float, int, len, print, range, enumerate, zip, map, filter, any, all, sorted)
BUILTIN_FUNCTIONS_MAP = {'abs': abs_, 'any': any_, 'all': all_, 'enumerate': enumerate_, 'filter': filter_, 'float': float_, 'int': int_, 'len': len_, 'map': map_, 'next': next_, 'print': print_, 'range': range_, 'sorted': sorted_, 'zip': zip_}