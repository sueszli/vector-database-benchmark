"""Python's built-in :mod:`functools` module builds several useful
utilities on top of Python's first-class function
support. ``funcutils`` generally stays in the same vein, adding to and
correcting Python's standard metaprogramming facilities.
"""
from __future__ import print_function
import sys
import re
import inspect
import functools
import itertools
from types import MethodType, FunctionType
try:
    xrange
    make_method = MethodType
except NameError:
    make_method = lambda desc, obj, obj_type: MethodType(desc, obj)
    basestring = (str, bytes)
    _IS_PY2 = False
else:
    _IS_PY2 = True
try:
    _inspect_iscoroutinefunction = inspect.iscoroutinefunction
except AttributeError:
    _inspect_iscoroutinefunction = lambda func: False
try:
    from .typeutils import make_sentinel
    NO_DEFAULT = make_sentinel(var_name='NO_DEFAULT')
except ImportError:
    NO_DEFAULT = object()
try:
    from functools import partialmethod
except ImportError:
    partialmethod = None
_IS_PY35 = sys.version_info >= (3, 5)
if not _IS_PY35:
    from inspect import formatargspec as inspect_formatargspec
else:
    from inspect import formatannotation

    def inspect_formatargspec(args, varargs=None, varkw=None, defaults=None, kwonlyargs=(), kwonlydefaults={}, annotations={}, formatarg=str, formatvarargs=lambda name: '*' + name, formatvarkw=lambda name: '**' + name, formatvalue=lambda value: '=' + repr(value), formatreturns=lambda text: ' -> ' + text, formatannotation=formatannotation):
        if False:
            return 10
        'Copy formatargspec from python 3.7 standard library.\n        Python 3 has deprecated formatargspec and requested that Signature\n        be used instead, however this requires a full reimplementation\n        of formatargspec() in terms of creating Parameter objects and such.\n        Instead of introducing all the object-creation overhead and having\n        to reinvent from scratch, just copy their compatibility routine.\n        '

        def formatargandannotation(arg):
            if False:
                return 10
            result = formatarg(arg)
            if arg in annotations:
                result += ': ' + formatannotation(annotations[arg])
            return result
        specs = []
        if defaults:
            firstdefault = len(args) - len(defaults)
        for (i, arg) in enumerate(args):
            spec = formatargandannotation(arg)
            if defaults and i >= firstdefault:
                spec = spec + formatvalue(defaults[i - firstdefault])
            specs.append(spec)
        if varargs is not None:
            specs.append(formatvarargs(formatargandannotation(varargs)))
        elif kwonlyargs:
            specs.append('*')
        if kwonlyargs:
            for kwonlyarg in kwonlyargs:
                spec = formatargandannotation(kwonlyarg)
                if kwonlydefaults and kwonlyarg in kwonlydefaults:
                    spec += formatvalue(kwonlydefaults[kwonlyarg])
                specs.append(spec)
        if varkw is not None:
            specs.append(formatvarkw(formatargandannotation(varkw)))
        result = '(' + ', '.join(specs) + ')'
        if 'return' in annotations:
            result += formatreturns(formatannotation(annotations['return']))
        return result

def get_module_callables(mod, ignore=None):
    if False:
        i = 10
        return i + 15
    'Returns two maps of (*types*, *funcs*) from *mod*, optionally\n    ignoring based on the :class:`bool` return value of the *ignore*\n    callable. *mod* can be a string name of a module in\n    :data:`sys.modules` or the module instance itself.\n    '
    if isinstance(mod, basestring):
        mod = sys.modules[mod]
    (types, funcs) = ({}, {})
    for attr_name in dir(mod):
        if ignore and ignore(attr_name):
            continue
        try:
            attr = getattr(mod, attr_name)
        except Exception:
            continue
        try:
            attr_mod_name = attr.__module__
        except AttributeError:
            continue
        if attr_mod_name != mod.__name__:
            continue
        if isinstance(attr, type):
            types[attr_name] = attr
        elif callable(attr):
            funcs[attr_name] = attr
    return (types, funcs)

def mro_items(type_obj):
    if False:
        i = 10
        return i + 15
    "Takes a type and returns an iterator over all class variables\n    throughout the type hierarchy (respecting the MRO).\n\n    >>> sorted(set([k for k, v in mro_items(int) if not k.startswith('__') and 'bytes' not in k and not callable(v)]))\n    ['denominator', 'imag', 'numerator', 'real']\n    "
    return itertools.chain.from_iterable((ct.__dict__.items() for ct in type_obj.__mro__))

def dir_dict(obj, raise_exc=False):
    if False:
        i = 10
        return i + 15
    'Return a dictionary of attribute names to values for a given\n    object. Unlike ``obj.__dict__``, this function returns all\n    attributes on the object, including ones on parent classes.\n    '
    ret = {}
    for k in dir(obj):
        try:
            ret[k] = getattr(obj, k)
        except Exception:
            if raise_exc:
                raise
    return ret

def copy_function(orig, copy_dict=True):
    if False:
        i = 10
        return i + 15
    'Returns a shallow copy of the function, including code object,\n    globals, closure, etc.\n\n    >>> func = lambda: func\n    >>> func() is func\n    True\n    >>> func_copy = copy_function(func)\n    >>> func_copy() is func\n    True\n    >>> func_copy is not func\n    True\n\n    Args:\n        orig (function): The function to be copied. Must be a\n            function, not just any method or callable.\n        copy_dict (bool): Also copy any attributes set on the function\n            instance. Defaults to ``True``.\n    '
    ret = FunctionType(orig.__code__, orig.__globals__, name=orig.__name__, argdefs=getattr(orig, '__defaults__', None), closure=getattr(orig, '__closure__', None))
    if hasattr(orig, '__kwdefaults__'):
        ret.__kwdefaults__ = orig.__kwdefaults__
    if copy_dict:
        ret.__dict__.update(orig.__dict__)
    return ret

def partial_ordering(cls):
    if False:
        i = 10
        return i + 15
    'Class decorator, similar to :func:`functools.total_ordering`,\n    except it is used to define `partial orderings`_ (i.e., it is\n    possible that *x* is neither greater than, equal to, or less than\n    *y*). It assumes the presence of the ``__le__()`` and ``__ge__()``\n    method, but nothing else. It will not override any existing\n    additional comparison methods.\n\n    .. _partial orderings: https://en.wikipedia.org/wiki/Partially_ordered_set\n\n    >>> @partial_ordering\n    ... class MySet(set):\n    ...     def __le__(self, other):\n    ...         return self.issubset(other)\n    ...     def __ge__(self, other):\n    ...         return self.issuperset(other)\n    ...\n    >>> a = MySet([1,2,3])\n    >>> b = MySet([1,2])\n    >>> c = MySet([1,2,4])\n    >>> b < a\n    True\n    >>> b > a\n    False\n    >>> b < c\n    True\n    >>> a < c\n    False\n    >>> c > a\n    False\n    '

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self <= other and (not self >= other)

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self >= other and (not self <= other)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self >= other and self <= other
    if not hasattr(cls, '__lt__'):
        cls.__lt__ = __lt__
    if not hasattr(cls, '__gt__'):
        cls.__gt__ = __gt__
    if not hasattr(cls, '__eq__'):
        cls.__eq__ = __eq__
    return cls

class InstancePartial(functools.partial):
    """:class:`functools.partial` is a huge convenience for anyone
    working with Python's great first-class functions. It allows
    developers to curry arguments and incrementally create simpler
    callables for a variety of use cases.

    Unfortunately there's one big gap in its usefulness:
    methods. Partials just don't get bound as methods and
    automatically handed a reference to ``self``. The
    ``InstancePartial`` type remedies this by inheriting from
    :class:`functools.partial` and implementing the necessary
    descriptor protocol. There are no other differences in
    implementation or usage. :class:`CachedInstancePartial`, below,
    has the same ability, but is slightly more efficient.

    """
    if partialmethod is not None:

        @property
        def _partialmethod(self):
            if False:
                for i in range(10):
                    print('nop')
            return partialmethod(self.func, *self.args, **self.keywords)

    def __get__(self, obj, obj_type):
        if False:
            return 10
        return make_method(self, obj, obj_type)

class CachedInstancePartial(functools.partial):
    """The ``CachedInstancePartial`` is virtually the same as
    :class:`InstancePartial`, adding support for method-usage to
    :class:`functools.partial`, except that upon first access, it
    caches the bound method on the associated object, speeding it up
    for future accesses, and bringing the method call overhead to
    about the same as non-``partial`` methods.

    See the :class:`InstancePartial` docstring for more details.
    """
    if partialmethod is not None:

        @property
        def _partialmethod(self):
            if False:
                return 10
            return partialmethod(self.func, *self.args, **self.keywords)
    if sys.version_info >= (3, 6):

        def __set_name__(self, obj_type, name):
            if False:
                print('Hello World!')
            self.__name__ = name

    def __get__(self, obj, obj_type):
        if False:
            for i in range(10):
                print('nop')
        self.__name__ = getattr(self, '__name__', None)
        self.__doc__ = self.func.__doc__
        self.__module__ = self.func.__module__
        name = self.__name__
        if name is None:
            for (k, v) in mro_items(obj_type):
                if v is self:
                    self.__name__ = name = k
        if obj is None:
            return make_method(self, obj, obj_type)
        try:
            return obj.__dict__[name]
        except KeyError:
            obj.__dict__[name] = ret = make_method(self, obj, obj_type)
            return ret
partial = CachedInstancePartial

def format_invocation(name='', args=(), kwargs=None, **kw):
    if False:
        return 10
    "Given a name, positional arguments, and keyword arguments, format\n    a basic Python-style function call.\n\n    >>> print(format_invocation('func', args=(1, 2), kwargs={'c': 3}))\n    func(1, 2, c=3)\n    >>> print(format_invocation('a_func', args=(1,)))\n    a_func(1)\n    >>> print(format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]))\n    kw_func(a=1, b=2)\n\n    "
    _repr = kw.pop('repr', repr)
    if kw:
        raise TypeError('unexpected keyword args: %r' % ', '.join(kw.keys()))
    kwargs = kwargs or {}
    a_text = ', '.join([_repr(a) for a in args])
    if isinstance(kwargs, dict):
        kwarg_items = [(k, kwargs[k]) for k in sorted(kwargs)]
    else:
        kwarg_items = kwargs
    kw_text = ', '.join(['%s=%s' % (k, _repr(v)) for (k, v) in kwarg_items])
    all_args_text = a_text
    if all_args_text and kw_text:
        all_args_text += ', '
    all_args_text += kw_text
    return '%s(%s)' % (name, all_args_text)

def format_exp_repr(obj, pos_names, req_names=None, opt_names=None, opt_key=None):
    if False:
        print('Hello World!')
    "Render an expression-style repr of an object, based on attribute\n    names, which are assumed to line up with arguments to an initializer.\n\n    >>> class Flag(object):\n    ...    def __init__(self, length, width, depth=None):\n    ...        self.length = length\n    ...        self.width = width\n    ...        self.depth = depth\n    ...\n\n    That's our Flag object, here are some example reprs for it:\n\n    >>> flag = Flag(5, 10)\n    >>> print(format_exp_repr(flag, ['length', 'width'], [], ['depth']))\n    Flag(5, 10)\n    >>> flag2 = Flag(5, 15, 2)\n    >>> print(format_exp_repr(flag2, ['length'], ['width', 'depth']))\n    Flag(5, width=15, depth=2)\n\n    By picking the pos_names, req_names, opt_names, and opt_key, you\n    can fine-tune how you want the repr to look.\n\n    Args:\n       obj (object): The object whose type name will be used and\n          attributes will be checked\n       pos_names (list): Required list of attribute names which will be\n          rendered as positional arguments in the output repr.\n       req_names (list): List of attribute names which will always\n          appear in the keyword arguments in the output repr. Defaults to None.\n       opt_names (list): List of attribute names which may appear in\n          the keyword arguments in the output repr, provided they pass\n          the *opt_key* check. Defaults to None.\n       opt_key (callable): A function or callable which checks whether\n          an opt_name should be in the repr. Defaults to a\n          ``None``-check.\n\n    "
    cn = type(obj).__name__
    req_names = req_names or []
    opt_names = opt_names or []
    (uniq_names, all_names) = (set(), [])
    for name in req_names + opt_names:
        if name in uniq_names:
            continue
        uniq_names.add(name)
        all_names.append(name)
    if opt_key is None:
        opt_key = lambda v: v is None
    assert callable(opt_key)
    args = [getattr(obj, name, None) for name in pos_names]
    kw_items = [(name, getattr(obj, name, None)) for name in all_names]
    kw_items = [(name, val) for (name, val) in kw_items if not (name in opt_names and opt_key(val))]
    return format_invocation(cn, args, kw_items)

def format_nonexp_repr(obj, req_names=None, opt_names=None, opt_key=None):
    if False:
        i = 10
        return i + 15
    "Format a non-expression-style repr\n\n    Some object reprs look like object instantiation, e.g., App(r=[], mw=[]).\n\n    This makes sense for smaller, lower-level objects whose state\n    roundtrips. But a lot of objects contain values that don't\n    roundtrip, like types and functions.\n\n    For those objects, there is the non-expression style repr, which\n    mimic's Python's default style to make a repr like so:\n\n    >>> class Flag(object):\n    ...    def __init__(self, length, width, depth=None):\n    ...        self.length = length\n    ...        self.width = width\n    ...        self.depth = depth\n    ...\n    >>> flag = Flag(5, 10)\n    >>> print(format_nonexp_repr(flag, ['length', 'width'], ['depth']))\n    <Flag length=5 width=10>\n\n    If no attributes are specified or set, utilizes the id, not unlike Python's\n    built-in behavior.\n\n    >>> print(format_nonexp_repr(flag))\n    <Flag id=...>\n    "
    cn = obj.__class__.__name__
    req_names = req_names or []
    opt_names = opt_names or []
    (uniq_names, all_names) = (set(), [])
    for name in req_names + opt_names:
        if name in uniq_names:
            continue
        uniq_names.add(name)
        all_names.append(name)
    if opt_key is None:
        opt_key = lambda v: v is None
    assert callable(opt_key)
    items = [(name, getattr(obj, name, None)) for name in all_names]
    labels = ['%s=%r' % (name, val) for (name, val) in items if not (name in opt_names and opt_key(val))]
    if not labels:
        labels = ['id=%s' % id(obj)]
    ret = '<%s %s>' % (cn, ' '.join(labels))
    return ret

def wraps(func, injected=None, expected=None, **kw):
    if False:
        for i in range(10):
            print('nop')
    "Decorator factory to apply update_wrapper() to a wrapper function.\n\n    Modeled after built-in :func:`functools.wraps`. Returns a decorator\n    that invokes update_wrapper() with the decorated function as the wrapper\n    argument and the arguments to wraps() as the remaining arguments.\n    Default arguments are as for update_wrapper(). This is a convenience\n    function to simplify applying partial() to update_wrapper().\n\n    Same example as in update_wrapper's doc but with wraps:\n\n        >>> from boltons.funcutils import wraps\n        >>>\n        >>> def print_return(func):\n        ...     @wraps(func)\n        ...     def wrapper(*args, **kwargs):\n        ...         ret = func(*args, **kwargs)\n        ...         print(ret)\n        ...         return ret\n        ...     return wrapper\n        ...\n        >>> @print_return\n        ... def example():\n        ...     '''docstring'''\n        ...     return 'example return value'\n        >>>\n        >>> val = example()\n        example return value\n        >>> example.__name__\n        'example'\n        >>> example.__doc__\n        'docstring'\n    "
    return partial(update_wrapper, func=func, build_from=None, injected=injected, expected=expected, **kw)

def update_wrapper(wrapper, func, injected=None, expected=None, build_from=None, **kw):
    if False:
        i = 10
        return i + 15
    "Modeled after the built-in :func:`functools.update_wrapper`,\n    this function is used to make your wrapper function reflect the\n    wrapped function's:\n\n      * Name\n      * Documentation\n      * Module\n      * Signature\n\n    The built-in :func:`functools.update_wrapper` copies the first three, but\n    does not copy the signature. This version of ``update_wrapper`` can copy\n    the inner function's signature exactly, allowing seamless usage\n    and :mod:`introspection <inspect>`. Usage is identical to the\n    built-in version::\n\n        >>> from boltons.funcutils import update_wrapper\n        >>>\n        >>> def print_return(func):\n        ...     def wrapper(*args, **kwargs):\n        ...         ret = func(*args, **kwargs)\n        ...         print(ret)\n        ...         return ret\n        ...     return update_wrapper(wrapper, func)\n        ...\n        >>> @print_return\n        ... def example():\n        ...     '''docstring'''\n        ...     return 'example return value'\n        >>>\n        >>> val = example()\n        example return value\n        >>> example.__name__\n        'example'\n        >>> example.__doc__\n        'docstring'\n\n    In addition, the boltons version of update_wrapper supports\n    modifying the outer signature. By passing a list of\n    *injected* argument names, those arguments will be removed from\n    the outer wrapper's signature, allowing your decorator to provide\n    arguments that aren't passed in.\n\n    Args:\n\n        wrapper (function) : The callable to which the attributes of\n            *func* are to be copied.\n        func (function): The callable whose attributes are to be copied.\n        injected (list): An optional list of argument names which\n            should not appear in the new wrapper's signature.\n        expected (list): An optional list of argument names (or (name,\n            default) pairs) representing new arguments introduced by\n            the wrapper (the opposite of *injected*). See\n            :meth:`FunctionBuilder.add_arg()` for more details.\n        build_from (function): The callable from which the new wrapper\n            is built. Defaults to *func*, unless *wrapper* is partial object\n            built from *func*, in which case it defaults to *wrapper*.\n            Useful in some specific cases where *wrapper* and *func* have the\n            same arguments but differ on which are keyword-only and positional-only.\n        update_dict (bool): Whether to copy other, non-standard\n            attributes of *func* over to the wrapper. Defaults to True.\n        inject_to_varkw (bool): Ignore missing arguments when a\n            ``**kwargs``-type catch-all is present. Defaults to True.\n        hide_wrapped (bool): Remove reference to the wrapped function(s)\n            in the updated function.\n\n    In opposition to the built-in :func:`functools.update_wrapper` bolton's\n    version returns a copy of the function and does not modify anything in place.\n    For more in-depth wrapping of functions, see the\n    :class:`FunctionBuilder` type, on which update_wrapper was built.\n    "
    if injected is None:
        injected = []
    elif isinstance(injected, basestring):
        injected = [injected]
    else:
        injected = list(injected)
    expected_items = _parse_wraps_expected(expected)
    if isinstance(func, (classmethod, staticmethod)):
        raise TypeError('wraps does not support wrapping classmethods and staticmethods, change the order of wrapping to wrap the underlying function: %r' % (getattr(func, '__func__', None),))
    update_dict = kw.pop('update_dict', True)
    inject_to_varkw = kw.pop('inject_to_varkw', True)
    hide_wrapped = kw.pop('hide_wrapped', False)
    if kw:
        raise TypeError('unexpected kwargs: %r' % kw.keys())
    if isinstance(wrapper, functools.partial) and func is wrapper.func:
        build_from = build_from or wrapper
    fb = FunctionBuilder.from_func(build_from or func)
    for arg in injected:
        try:
            fb.remove_arg(arg)
        except MissingArgument:
            if inject_to_varkw and fb.varkw is not None:
                continue
            raise
    for (arg, default) in expected_items:
        fb.add_arg(arg, default)
    if fb.is_async:
        fb.body = 'return await _call(%s)' % fb.get_invocation_str()
    else:
        fb.body = 'return _call(%s)' % fb.get_invocation_str()
    execdict = dict(_call=wrapper, _func=func)
    fully_wrapped = fb.get_func(execdict, with_dict=update_dict)
    if hide_wrapped and hasattr(fully_wrapped, '__wrapped__'):
        del fully_wrapped.__dict__['__wrapped__']
    elif not hide_wrapped:
        fully_wrapped.__wrapped__ = func
    return fully_wrapped

def _parse_wraps_expected(expected):
    if False:
        while True:
            i = 10
    if expected is None:
        expected = []
    elif isinstance(expected, basestring):
        expected = [(expected, NO_DEFAULT)]
    expected_items = []
    try:
        expected_iter = iter(expected)
    except TypeError as e:
        raise ValueError('"expected" takes string name, sequence of string names, iterable of (name, default) pairs, or a mapping of  {name: default}, not %r (got: %r)' % (expected, e))
    for argname in expected_iter:
        if isinstance(argname, basestring):
            try:
                default = expected[argname]
            except TypeError:
                default = NO_DEFAULT
        else:
            try:
                (argname, default) = argname
            except (TypeError, ValueError):
                raise ValueError('"expected" takes string name, sequence of string names, iterable of (name, default) pairs, or a mapping of  {name: default}, not %r')
        if not isinstance(argname, basestring):
            raise ValueError('all "expected" argnames must be strings, not %r' % (argname,))
        expected_items.append((argname, default))
    return expected_items

class FunctionBuilder(object):
    """The FunctionBuilder type provides an interface for programmatically
    creating new functions, either based on existing functions or from
    scratch.

    Values are passed in at construction or set as attributes on the
    instance. For creating a new function based of an existing one,
    see the :meth:`~FunctionBuilder.from_func` classmethod. At any
    point, :meth:`~FunctionBuilder.get_func` can be called to get a
    newly compiled function, based on the values configured.

    >>> fb = FunctionBuilder('return_five', doc='returns the integer 5',
    ...                      body='return 5')
    >>> f = fb.get_func()
    >>> f()
    5
    >>> fb.varkw = 'kw'
    >>> f_kw = fb.get_func()
    >>> f_kw(ignored_arg='ignored_val')
    5

    Note that function signatures themselves changed quite a bit in
    Python 3, so several arguments are only applicable to
    FunctionBuilder in Python 3. Except for *name*, all arguments to
    the constructor are keyword arguments.

    Args:
        name (str): Name of the function.
        doc (str): `Docstring`_ for the function, defaults to empty.
        module (str): Name of the module from which this function was
            imported. Defaults to None.
        body (str): String version of the code representing the body
            of the function. Defaults to ``'pass'``, which will result
            in a function which does nothing and returns ``None``.
        args (list): List of argument names, defaults to empty list,
            denoting no arguments.
        varargs (str): Name of the catch-all variable for positional
            arguments. E.g., "args" if the resultant function is to have
            ``*args`` in the signature. Defaults to None.
        varkw (str): Name of the catch-all variable for keyword
            arguments. E.g., "kwargs" if the resultant function is to have
            ``**kwargs`` in the signature. Defaults to None.
        defaults (tuple): A tuple containing default argument values for
            those arguments that have defaults.
        kwonlyargs (list): Argument names which are only valid as
            keyword arguments. **Python 3 only.**
        kwonlydefaults (dict): A mapping, same as normal *defaults*,
            but only for the *kwonlyargs*. **Python 3 only.**
        annotations (dict): Mapping of type hints and so
            forth. **Python 3 only.**
        filename (str): The filename that will appear in
            tracebacks. Defaults to "boltons.funcutils.FunctionBuilder".
        indent (int): Number of spaces with which to indent the
            function *body*. Values less than 1 will result in an error.
        dict (dict): Any other attributes which should be added to the
            functions compiled with this FunctionBuilder.

    All of these arguments are also made available as attributes which
    can be mutated as necessary.

    .. _Docstring: https://en.wikipedia.org/wiki/Docstring#Python

    """
    if _IS_PY2:
        _argspec_defaults = {'args': list, 'varargs': lambda : None, 'varkw': lambda : None, 'defaults': lambda : None}

        @classmethod
        def _argspec_to_dict(cls, f):
            if False:
                while True:
                    i = 10
            (args, varargs, varkw, defaults) = inspect.getargspec(f)
            return {'args': args, 'varargs': varargs, 'varkw': varkw, 'defaults': defaults}
    else:
        _argspec_defaults = {'args': list, 'varargs': lambda : None, 'varkw': lambda : None, 'defaults': lambda : None, 'kwonlyargs': list, 'kwonlydefaults': dict, 'annotations': dict}

        @classmethod
        def _argspec_to_dict(cls, f):
            if False:
                return 10
            argspec = inspect.getfullargspec(f)
            return dict(((attr, getattr(argspec, attr)) for attr in cls._argspec_defaults))
    _defaults = {'doc': str, 'dict': dict, 'is_async': lambda : False, 'module': lambda : None, 'body': lambda : 'pass', 'indent': lambda : 4, 'annotations': dict, 'filename': lambda : 'boltons.funcutils.FunctionBuilder'}
    _defaults.update(_argspec_defaults)
    _compile_count = itertools.count()

    def __init__(self, name, **kw):
        if False:
            while True:
                i = 10
        self.name = name
        for (a, default_factory) in self._defaults.items():
            val = kw.pop(a, None)
            if val is None:
                val = default_factory()
            setattr(self, a, val)
        if kw:
            raise TypeError('unexpected kwargs: %r' % kw.keys())
        return
    if _IS_PY2:

        def get_sig_str(self, with_annotations=True):
            if False:
                for i in range(10):
                    print('nop')
            'Return function signature as a string.\n\n            with_annotations is ignored on Python 2.  On Python 3 signature\n            will omit annotations if it is set to False.\n            '
            return inspect_formatargspec(self.args, self.varargs, self.varkw, [])

        def get_invocation_str(self):
            if False:
                return 10
            return inspect_formatargspec(self.args, self.varargs, self.varkw, [])[1:-1]
    else:

        def get_sig_str(self, with_annotations=True):
            if False:
                print('Hello World!')
            'Return function signature as a string.\n\n            with_annotations is ignored on Python 2.  On Python 3 signature\n            will omit annotations if it is set to False.\n            '
            if with_annotations:
                annotations = self.annotations
            else:
                annotations = {}
            return inspect_formatargspec(self.args, self.varargs, self.varkw, [], self.kwonlyargs, {}, annotations)
        _KWONLY_MARKER = re.compile('\n        \\*     # a star\n        \\s*    # followed by any amount of whitespace\n        ,      # followed by a comma\n        \\s*    # followed by any amount of whitespace\n        ', re.VERBOSE)

        def get_invocation_str(self):
            if False:
                for i in range(10):
                    print('nop')
            kwonly_pairs = None
            formatters = {}
            if self.kwonlyargs:
                kwonly_pairs = dict(((arg, arg) for arg in self.kwonlyargs))
                formatters['formatvalue'] = lambda value: '=' + value
            sig = inspect_formatargspec(self.args, self.varargs, self.varkw, [], kwonly_pairs, kwonly_pairs, {}, **formatters)
            sig = self._KWONLY_MARKER.sub('', sig)
            return sig[1:-1]

    @classmethod
    def from_func(cls, func):
        if False:
            return 10
        'Create a new FunctionBuilder instance based on an existing\n        function. The original function will not be stored or\n        modified.\n        '
        if not callable(func):
            raise TypeError('expected callable object, not %r' % (func,))
        if isinstance(func, functools.partial):
            if _IS_PY2:
                raise ValueError('Cannot build FunctionBuilder instances from partials in python 2.')
            kwargs = {'name': func.func.__name__, 'doc': func.func.__doc__, 'module': getattr(func.func, '__module__', None), 'annotations': getattr(func.func, '__annotations__', {}), 'dict': getattr(func.func, '__dict__', {})}
        else:
            kwargs = {'name': func.__name__, 'doc': func.__doc__, 'module': getattr(func, '__module__', None), 'annotations': getattr(func, '__annotations__', {}), 'dict': getattr(func, '__dict__', {})}
        kwargs.update(cls._argspec_to_dict(func))
        if _inspect_iscoroutinefunction(func):
            kwargs['is_async'] = True
        return cls(**kwargs)

    def get_func(self, execdict=None, add_source=True, with_dict=True):
        if False:
            while True:
                i = 10
        'Compile and return a new function based on the current values of\n        the FunctionBuilder.\n\n        Args:\n            execdict (dict): The dictionary representing the scope in\n                which the compilation should take place. Defaults to an empty\n                dict.\n            add_source (bool): Whether to add the source used to a\n                special ``__source__`` attribute on the resulting\n                function. Defaults to True.\n            with_dict (bool): Add any custom attributes, if\n                applicable. Defaults to True.\n\n        To see an example of usage, see the implementation of\n        :func:`~boltons.funcutils.wraps`.\n        '
        execdict = execdict or {}
        body = self.body or self._default_body
        tmpl = 'def {name}{sig_str}:'
        tmpl += '\n{body}'
        if self.is_async:
            tmpl = 'async ' + tmpl
        body = _indent(self.body, ' ' * self.indent)
        name = self.name.replace('<', '_').replace('>', '_')
        src = tmpl.format(name=name, sig_str=self.get_sig_str(with_annotations=False), doc=self.doc, body=body)
        self._compile(src, execdict)
        func = execdict[name]
        func.__name__ = self.name
        func.__doc__ = self.doc
        func.__defaults__ = self.defaults
        if not _IS_PY2:
            func.__kwdefaults__ = self.kwonlydefaults
            func.__annotations__ = self.annotations
        if with_dict:
            func.__dict__.update(self.dict)
        func.__module__ = self.module
        if add_source:
            func.__source__ = src
        return func

    def get_defaults_dict(self):
        if False:
            i = 10
            return i + 15
        'Get a dictionary of function arguments with defaults and the\n        respective values.\n        '
        ret = dict(reversed(list(zip(reversed(self.args), reversed(self.defaults or [])))))
        kwonlydefaults = getattr(self, 'kwonlydefaults', None)
        if kwonlydefaults:
            ret.update(kwonlydefaults)
        return ret

    def get_arg_names(self, only_required=False):
        if False:
            i = 10
            return i + 15
        arg_names = tuple(self.args) + tuple(getattr(self, 'kwonlyargs', ()))
        if only_required:
            defaults_dict = self.get_defaults_dict()
            arg_names = tuple([an for an in arg_names if an not in defaults_dict])
        return arg_names
    if _IS_PY2:

        def add_arg(self, arg_name, default=NO_DEFAULT):
            if False:
                return 10
            'Add an argument with optional *default* (defaults to ``funcutils.NO_DEFAULT``).'
            if arg_name in self.args:
                raise ExistingArgument('arg %r already in func %s arg list' % (arg_name, self.name))
            self.args.append(arg_name)
            if default is not NO_DEFAULT:
                self.defaults = (self.defaults or ()) + (default,)
            return
    else:

        def add_arg(self, arg_name, default=NO_DEFAULT, kwonly=False):
            if False:
                print('Hello World!')
            'Add an argument with optional *default* (defaults to\n            ``funcutils.NO_DEFAULT``). Pass *kwonly=True* to add a\n            keyword-only argument\n            '
            if arg_name in self.args:
                raise ExistingArgument('arg %r already in func %s arg list' % (arg_name, self.name))
            if arg_name in self.kwonlyargs:
                raise ExistingArgument('arg %r already in func %s kwonly arg list' % (arg_name, self.name))
            if not kwonly:
                self.args.append(arg_name)
                if default is not NO_DEFAULT:
                    self.defaults = (self.defaults or ()) + (default,)
            else:
                self.kwonlyargs.append(arg_name)
                if default is not NO_DEFAULT:
                    self.kwonlydefaults[arg_name] = default
            return

    def remove_arg(self, arg_name):
        if False:
            return 10
        "Remove an argument from this FunctionBuilder's argument list. The\n        resulting function will have one less argument per call to\n        this function.\n\n        Args:\n            arg_name (str): The name of the argument to remove.\n\n        Raises a :exc:`ValueError` if the argument is not present.\n\n        "
        args = self.args
        d_dict = self.get_defaults_dict()
        try:
            args.remove(arg_name)
        except ValueError:
            try:
                self.kwonlyargs.remove(arg_name)
            except (AttributeError, ValueError):
                exc = MissingArgument('arg %r not found in %s argument list: %r' % (arg_name, self.name, args))
                exc.arg_name = arg_name
                raise exc
            else:
                self.kwonlydefaults.pop(arg_name, None)
        else:
            d_dict.pop(arg_name, None)
            self.defaults = tuple([d_dict[a] for a in args if a in d_dict])
        return

    def _compile(self, src, execdict):
        if False:
            while True:
                i = 10
        filename = '<%s-%d>' % (self.filename, next(self._compile_count))
        try:
            code = compile(src, filename, 'single')
            exec(code, execdict)
        except Exception:
            raise
        return execdict

class MissingArgument(ValueError):
    pass

class ExistingArgument(ValueError):
    pass

def _indent(text, margin, newline='\n', key=bool):
    if False:
        i = 10
        return i + 15
    'based on boltons.strutils.indent'
    indented_lines = [margin + line if key(line) else line for line in text.splitlines()]
    return newline.join(indented_lines)
try:
    from functools import total_ordering
except ImportError:

    def total_ordering(cls):
        if False:
            print('Hello World!')
        'Class decorator that fills in missing comparators/ordering\n        methods. Backport of :func:`functools.total_ordering` to work\n        with Python 2.6.\n\n        Code from http://code.activestate.com/recipes/576685/\n        '
        convert = {'__lt__': [('__gt__', lambda self, other: not (self < other or self == other)), ('__le__', lambda self, other: self < other or self == other), ('__ge__', lambda self, other: not self < other)], '__le__': [('__ge__', lambda self, other: not self <= other or self == other), ('__lt__', lambda self, other: self <= other and (not self == other)), ('__gt__', lambda self, other: not self <= other)], '__gt__': [('__lt__', lambda self, other: not (self > other or self == other)), ('__ge__', lambda self, other: self > other or self == other), ('__le__', lambda self, other: not self > other)], '__ge__': [('__le__', lambda self, other: not self >= other or self == other), ('__gt__', lambda self, other: self >= other and (not self == other)), ('__lt__', lambda self, other: not self >= other)]}
        roots = set(dir(cls)) & set(convert)
        if not roots:
            raise ValueError('must define at least one ordering operation: < > <= >=')
        root = max(roots)
        for (opname, opfunc) in convert[root]:
            if opname not in roots:
                opfunc.__name__ = opname
                opfunc.__doc__ = getattr(int, opname).__doc__
                setattr(cls, opname, opfunc)
        return cls

def noop(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Simple function that should be used when no effect is desired.\n    An alternative to checking for  an optional function type parameter.\n\n    e.g.\n    def decorate(func, pre_func=None, post_func=None):\n        if pre_func:\n            pre_func()\n        func()\n        if post_func:\n            post_func()\n\n    vs\n\n    def decorate(func, pre_func=noop, post_func=noop):\n        pre_func()\n        func()\n        post_func()\n    '
    return None