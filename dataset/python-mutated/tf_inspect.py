"""TFDecorator-aware replacements for the inspect module."""
import collections
import functools
import inspect as _inspect
import six
from tensorflow.python.util import tf_decorator

def signature(obj, *, follow_wrapped=True):
    if False:
        return 10
    'TFDecorator-aware replacement for inspect.signature.'
    return _inspect.signature(tf_decorator.unwrap(obj)[1], follow_wrapped=follow_wrapped)
Parameter = _inspect.Parameter
Signature = _inspect.Signature
if hasattr(_inspect, 'ArgSpec'):
    ArgSpec = _inspect.ArgSpec
else:
    ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
if hasattr(_inspect, 'FullArgSpec'):
    FullArgSpec = _inspect.FullArgSpec
else:
    FullArgSpec = collections.namedtuple('FullArgSpec', ['args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults', 'annotations'])

def _convert_maybe_argspec_to_fullargspec(argspec):
    if False:
        return 10
    if isinstance(argspec, FullArgSpec):
        return argspec
    return FullArgSpec(args=argspec.args, varargs=argspec.varargs, varkw=argspec.keywords, defaults=argspec.defaults, kwonlyargs=[], kwonlydefaults=None, annotations={})
if hasattr(_inspect, 'getfullargspec'):
    _getfullargspec = _inspect.getfullargspec

    def _getargspec(target):
        if False:
            print('Hello World!')
        "A python3 version of getargspec.\n\n    Calls `getfullargspec` and assigns args, varargs,\n    varkw, and defaults to a python 2/3 compatible `ArgSpec`.\n\n    The parameter name 'varkw' is changed to 'keywords' to fit the\n    `ArgSpec` struct.\n\n    Args:\n      target: the target object to inspect.\n\n    Returns:\n      An ArgSpec with args, varargs, keywords, and defaults parameters\n      from FullArgSpec.\n    "
        fullargspecs = getfullargspec(target)
        defaults = fullargspecs.defaults or ()
        if fullargspecs.kwonlydefaults:
            defaults += tuple(fullargspecs.kwonlydefaults.values())
        if not defaults:
            defaults = None
        argspecs = ArgSpec(args=fullargspecs.args + fullargspecs.kwonlyargs, varargs=fullargspecs.varargs, keywords=fullargspecs.varkw, defaults=defaults)
        return argspecs
else:
    _getargspec = _inspect.getargspec

    def _getfullargspec(target):
        if False:
            while True:
                i = 10
        'A python2 version of getfullargspec.\n\n    Args:\n      target: the target object to inspect.\n\n    Returns:\n      A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.\n    '
        return _convert_maybe_argspec_to_fullargspec(getargspec(target))

def currentframe():
    if False:
        print('Hello World!')
    'TFDecorator-aware replacement for inspect.currentframe.'
    return _inspect.stack()[1][0]

def getargspec(obj):
    if False:
        for i in range(10):
            print('nop')
    "TFDecorator-aware replacement for `inspect.getargspec`.\n\n  Note: `getfullargspec` is recommended as the python 2/3 compatible\n  replacement for this function.\n\n  Args:\n    obj: A function, partial function, or callable object, possibly decorated.\n\n  Returns:\n    The `ArgSpec` that describes the signature of the outermost decorator that\n    changes the callable's signature, or the `ArgSpec` that describes\n    the object if not decorated.\n\n  Raises:\n    ValueError: When callable's signature can not be expressed with\n      ArgSpec.\n    TypeError: For objects of unsupported types.\n  "
    if isinstance(obj, functools.partial):
        return _get_argspec_for_partial(obj)
    (decorators, target) = tf_decorator.unwrap(obj)
    spec = next((d.decorator_argspec for d in decorators if d.decorator_argspec is not None), None)
    if spec:
        return spec
    try:
        return _getargspec(target)
    except TypeError:
        pass
    if isinstance(target, type):
        try:
            return _getargspec(target.__init__)
        except TypeError:
            pass
        try:
            return _getargspec(target.__new__)
        except TypeError:
            pass
    return _getargspec(type(target).__call__)

def _get_argspec_for_partial(obj):
    if False:
        while True:
            i = 10
    "Implements `getargspec` for `functools.partial` objects.\n\n  Args:\n    obj: The `functools.partial` object\n  Returns:\n    An `inspect.ArgSpec`\n  Raises:\n    ValueError: When callable's signature can not be expressed with\n      ArgSpec.\n  "
    n_prune_args = len(obj.args)
    partial_keywords = obj.keywords or {}
    (args, varargs, keywords, defaults) = getargspec(obj.func)
    args = args[n_prune_args:]
    no_default = object()
    all_defaults = [no_default] * len(args)
    if defaults:
        all_defaults[-len(defaults):] = defaults
    for (kw, default) in six.iteritems(partial_keywords):
        if kw in args:
            idx = args.index(kw)
            all_defaults[idx] = default
        elif not keywords:
            raise ValueError(f'{obj} does not have a **kwargs parameter, but contains an unknown partial keyword {kw}.')
    first_default = next((idx for (idx, x) in enumerate(all_defaults) if x is not no_default), None)
    if first_default is None:
        return ArgSpec(args, varargs, keywords, None)
    invalid_default_values = [args[i] for (i, j) in enumerate(all_defaults) if j is no_default and i > first_default]
    if invalid_default_values:
        raise ValueError(f'{obj} has some keyword-only arguments, which are not supported: {invalid_default_values}.')
    return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))

def getfullargspec(obj):
    if False:
        for i in range(10):
            print('nop')
    "TFDecorator-aware replacement for `inspect.getfullargspec`.\n\n  This wrapper emulates `inspect.getfullargspec` in[^)]* Python2.\n\n  Args:\n    obj: A callable, possibly decorated.\n\n  Returns:\n    The `FullArgSpec` that describes the signature of\n    the outermost decorator that changes the callable's signature. If the\n    callable is not decorated, `inspect.getfullargspec()` will be called\n    directly on the callable.\n  "
    (decorators, target) = tf_decorator.unwrap(obj)
    for d in decorators:
        if d.decorator_argspec is not None:
            return _convert_maybe_argspec_to_fullargspec(d.decorator_argspec)
    return _getfullargspec(target)

def getcallargs(*func_and_positional, **named):
    if False:
        return 10
    "TFDecorator-aware replacement for inspect.getcallargs.\n\n  Args:\n    *func_and_positional: A callable, possibly decorated, followed by any\n      positional arguments that would be passed to `func`.\n    **named: The named argument dictionary that would be passed to `func`.\n\n  Returns:\n    A dictionary mapping `func`'s named arguments to the values they would\n    receive if `func(*positional, **named)` were called.\n\n  `getcallargs` will use the argspec from the outermost decorator that provides\n  it. If no attached decorators modify argspec, the final unwrapped target's\n  argspec will be used.\n  "
    func = func_and_positional[0]
    positional = func_and_positional[1:]
    argspec = getfullargspec(func)
    call_args = named.copy()
    this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)
    if ismethod(func) and this:
        positional = (this,) + positional
    remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
    call_args.update(dict(zip(remaining_positionals, positional)))
    default_count = 0 if not argspec.defaults else len(argspec.defaults)
    if default_count:
        for (arg, value) in zip(argspec.args[-default_count:], argspec.defaults):
            if arg not in call_args:
                call_args[arg] = value
    if argspec.kwonlydefaults is not None:
        for (k, v) in argspec.kwonlydefaults.items():
            if k not in call_args:
                call_args[k] = v
    return call_args

def getframeinfo(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return _inspect.getframeinfo(*args, **kwargs)

def getdoc(object):
    if False:
        i = 10
        return i + 15
    'TFDecorator-aware replacement for inspect.getdoc.\n\n  Args:\n    object: An object, possibly decorated.\n\n  Returns:\n    The docstring associated with the object.\n\n  The outermost-decorated object is intended to have the most complete\n  documentation, so the decorated parameter is not unwrapped.\n  '
    return _inspect.getdoc(object)

def getfile(object):
    if False:
        return 10
    'TFDecorator-aware replacement for inspect.getfile.'
    unwrapped_object = tf_decorator.unwrap(object)[1]
    if hasattr(unwrapped_object, 'f_globals') and '__file__' in unwrapped_object.f_globals:
        return unwrapped_object.f_globals['__file__']
    return _inspect.getfile(unwrapped_object)

def getmembers(object, predicate=None):
    if False:
        print('Hello World!')
    'TFDecorator-aware replacement for inspect.getmembers.'
    return _inspect.getmembers(object, predicate)

def getmodule(object):
    if False:
        for i in range(10):
            print('nop')
    'TFDecorator-aware replacement for inspect.getmodule.'
    return _inspect.getmodule(object)

def getmro(cls):
    if False:
        while True:
            i = 10
    'TFDecorator-aware replacement for inspect.getmro.'
    return _inspect.getmro(cls)

def getsource(object):
    if False:
        while True:
            i = 10
    'TFDecorator-aware replacement for inspect.getsource.'
    return _inspect.getsource(tf_decorator.unwrap(object)[1])

def getsourcefile(object):
    if False:
        while True:
            i = 10
    'TFDecorator-aware replacement for inspect.getsourcefile.'
    return _inspect.getsourcefile(tf_decorator.unwrap(object)[1])

def getsourcelines(object):
    if False:
        print('Hello World!')
    'TFDecorator-aware replacement for inspect.getsourcelines.'
    return _inspect.getsourcelines(tf_decorator.unwrap(object)[1])

def isbuiltin(object):
    if False:
        for i in range(10):
            print('nop')
    'TFDecorator-aware replacement for inspect.isbuiltin.'
    return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])

def isclass(object):
    if False:
        for i in range(10):
            print('nop')
    'TFDecorator-aware replacement for inspect.isclass.'
    return _inspect.isclass(tf_decorator.unwrap(object)[1])

def isfunction(object):
    if False:
        i = 10
        return i + 15
    'TFDecorator-aware replacement for inspect.isfunction.'
    return _inspect.isfunction(tf_decorator.unwrap(object)[1])

def isframe(object):
    if False:
        print('Hello World!')
    'TFDecorator-aware replacement for inspect.ismodule.'
    return _inspect.isframe(tf_decorator.unwrap(object)[1])

def isgenerator(object):
    if False:
        print('Hello World!')
    'TFDecorator-aware replacement for inspect.isgenerator.'
    return _inspect.isgenerator(tf_decorator.unwrap(object)[1])

def isgeneratorfunction(object):
    if False:
        for i in range(10):
            print('nop')
    'TFDecorator-aware replacement for inspect.isgeneratorfunction.'
    return _inspect.isgeneratorfunction(tf_decorator.unwrap(object)[1])

def ismethod(object):
    if False:
        return 10
    'TFDecorator-aware replacement for inspect.ismethod.'
    return _inspect.ismethod(tf_decorator.unwrap(object)[1])

def isanytargetmethod(object):
    if False:
        for i in range(10):
            print('nop')
    'Checks if `object` or a TF Decorator wrapped target contains self or cls.\n\n  This function could be used along with `tf_inspect.getfullargspec` to\n  determine if the first argument of `object` argspec is self or cls. If the\n  first argument is self or cls, it needs to be excluded from argspec when we\n  compare the argspec to the input arguments and, if provided, the tf.function\n  input_signature.\n\n  Like `tf_inspect.getfullargspec` and python `inspect.getfullargspec`, it\n  does not unwrap python decorators.\n\n  Args:\n    obj: An method, function, or functool.partial, possibly decorated by\n    TFDecorator.\n\n  Returns:\n    A bool indicates if `object` or any target along the chain of TF decorators\n    is a method.\n  '
    (decorators, target) = tf_decorator.unwrap(object)
    for decorator in decorators:
        if _inspect.ismethod(decorator.decorated_target):
            return True
    while isinstance(target, functools.partial):
        target = target.func
    return callable(target) and (not _inspect.isfunction(target))

def ismodule(object):
    if False:
        i = 10
        return i + 15
    'TFDecorator-aware replacement for inspect.ismodule.'
    return _inspect.ismodule(tf_decorator.unwrap(object)[1])

def isroutine(object):
    if False:
        for i in range(10):
            print('nop')
    'TFDecorator-aware replacement for inspect.isroutine.'
    return _inspect.isroutine(tf_decorator.unwrap(object)[1])

def stack(context=1):
    if False:
        for i in range(10):
            print('nop')
    'TFDecorator-aware replacement for inspect.stack.'
    return _inspect.stack(context)[1:]