from datetime import tzinfo
from functools import partial
from operator import attrgetter
from numpy import dtype
import pandas as pd
from pytz import timezone
from six import iteritems, string_types, PY3
from toolz import valmap, complement, compose
import toolz.curried.operator as op
from zipline.utils.compat import wraps
from zipline.utils.functional import getattrs
from zipline.utils.preprocess import call, preprocess
if PY3:
    _qualified_name = attrgetter('__qualname__')
else:

    def _qualified_name(obj):
        if False:
            while True:
                i = 10
        '\n        Return the fully-qualified name (ignoring inner classes) of a type.\n        '
        try:
            return getattr(obj, '__qualname__')
        except AttributeError:
            pass
        module = obj.__module__
        if module in ('__builtin__', '__main__', 'builtins'):
            return obj.__name__
        return '.'.join([module, obj.__name__])

def verify_indices_all_unique(obj):
    if False:
        while True:
            i = 10
    '\n    Check that all axes of a pandas object are unique.\n\n    Parameters\n    ----------\n    obj : pd.Series / pd.DataFrame / pd.Panel\n        The object to validate.\n\n    Returns\n    -------\n    obj : pd.Series / pd.DataFrame / pd.Panel\n        The validated object, unchanged.\n\n    Raises\n    ------\n    ValueError\n        If any axis has duplicate entries.\n    '
    axis_names = [('index',), ('index', 'columns'), ('items', 'major_axis', 'minor_axis')][obj.ndim - 1]
    for (axis_name, index) in zip(axis_names, obj.axes):
        if index.is_unique:
            continue
        raise ValueError('Duplicate entries in {type}.{axis}: {dupes}.'.format(type=type(obj).__name__, axis=axis_name, dupes=sorted(index[index.duplicated()])))
    return obj

def optionally(preprocessor):
    if False:
        return 10
    "Modify a preprocessor to explicitly allow `None`.\n\n    Parameters\n    ----------\n    preprocessor : callable[callable, str, any -> any]\n        A preprocessor to delegate to when `arg is not None`.\n\n    Returns\n    -------\n    optional_preprocessor : callable[callable, str, any -> any]\n        A preprocessor that delegates to `preprocessor` when `arg is not None`.\n\n    Examples\n    --------\n    >>> def preprocessor(func, argname, arg):\n    ...     if not isinstance(arg, int):\n    ...         raise TypeError('arg must be int')\n    ...     return arg\n    ...\n    >>> @preprocess(a=optionally(preprocessor))\n    ... def f(a):\n    ...     return a\n    ...\n    >>> f(1)  # call with int\n    1\n    >>> f('a')  # call with not int\n    Traceback (most recent call last):\n       ...\n    TypeError: arg must be int\n    >>> f(None) is None  # call with explicit None\n    True\n    "

    @wraps(preprocessor)
    def wrapper(func, argname, arg):
        if False:
            i = 10
            return i + 15
        return arg if arg is None else preprocessor(func, argname, arg)
    return wrapper

def ensure_upper_case(func, argname, arg):
    if False:
        print('Hello World!')
    if isinstance(arg, string_types):
        return arg.upper()
    else:
        raise TypeError("{0}() expected argument '{1}' to be a string, but got {2} instead.".format(func.__name__, argname, arg))

def ensure_dtype(func, argname, arg):
    if False:
        print('Hello World!')
    "\n    Argument preprocessor that converts the input into a numpy dtype.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from zipline.utils.preprocess import preprocess\n    >>> @preprocess(dtype=ensure_dtype)\n    ... def foo(dtype):\n    ...     return dtype\n    ...\n    >>> foo(float)\n    dtype('float64')\n    "
    try:
        return dtype(arg)
    except TypeError:
        raise TypeError("{func}() couldn't convert argument {argname}={arg!r} to a numpy dtype.".format(func=_qualified_name(func), argname=argname, arg=arg))

def ensure_timezone(func, argname, arg):
    if False:
        while True:
            i = 10
    "Argument preprocessor that converts the input into a tzinfo object.\n\n    Examples\n    --------\n    >>> from zipline.utils.preprocess import preprocess\n    >>> @preprocess(tz=ensure_timezone)\n    ... def foo(tz):\n    ...     return tz\n    >>> foo('utc')\n    <UTC>\n    "
    if isinstance(arg, tzinfo):
        return arg
    if isinstance(arg, string_types):
        return timezone(arg)
    raise TypeError("{func}() couldn't convert argument {argname}={arg!r} to a timezone.".format(func=_qualified_name(func), argname=argname, arg=arg))

def ensure_timestamp(func, argname, arg):
    if False:
        print('Hello World!')
    "Argument preprocessor that converts the input into a pandas Timestamp\n    object.\n\n    Examples\n    --------\n    >>> from zipline.utils.preprocess import preprocess\n    >>> @preprocess(ts=ensure_timestamp)\n    ... def foo(ts):\n    ...     return ts\n    >>> foo('2014-01-01')\n    Timestamp('2014-01-01 00:00:00')\n    "
    try:
        return pd.Timestamp(arg)
    except ValueError as e:
        raise TypeError("{func}() couldn't convert argument {argname}={arg!r} to a pandas Timestamp.\nOriginal error was: {t}: {e}".format(func=_qualified_name(func), argname=argname, arg=arg, t=_qualified_name(type(e)), e=e))

def expect_dtypes(__funcname=_qualified_name, **named):
    if False:
        print('Hello World!')
    "\n    Preprocessing decorator that verifies inputs have expected numpy dtypes.\n\n    Examples\n    --------\n    >>> from numpy import dtype, arange, int8, float64\n    >>> @expect_dtypes(x=dtype(int8))\n    ... def foo(x, y):\n    ...    return x, y\n    ...\n    >>> foo(arange(3, dtype=int8), 'foo')\n    (array([0, 1, 2], dtype=int8), 'foo')\n    >>> foo(arange(3, dtype=float64), 'foo')  # doctest: +NORMALIZE_WHITESPACE\n    ...                                       # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    TypeError: ...foo() expected a value with dtype 'int8' for argument 'x',\n    but got 'float64' instead.\n    "
    for (name, type_) in iteritems(named):
        if not isinstance(type_, (dtype, tuple)):
            raise TypeError('expect_dtypes() expected a numpy dtype or tuple of dtypes for argument {name!r}, but got {dtype} instead.'.format(name=name, dtype=dtype))
    if isinstance(__funcname, str):

        def get_funcname(_):
            if False:
                print('Hello World!')
            return __funcname
    else:
        get_funcname = __funcname

    @preprocess(dtypes=call(lambda x: x if isinstance(x, tuple) else (x,)))
    def _expect_dtype(dtypes):
        if False:
            i = 10
            return i + 15
        '\n        Factory for dtype-checking functions that work with the @preprocess\n        decorator.\n        '

        def error_message(func, argname, value):
            if False:
                return 10
            try:
                value_to_show = value.dtype.name
            except AttributeError:
                value_to_show = value
            return '{funcname}() expected a value with dtype {dtype_str} for argument {argname!r}, but got {value!r} instead.'.format(funcname=get_funcname(func), dtype_str=' or '.join((repr(d.name) for d in dtypes)), argname=argname, value=value_to_show)

        def _actual_preprocessor(func, argname, argvalue):
            if False:
                for i in range(10):
                    print('nop')
            if getattr(argvalue, 'dtype', object()) not in dtypes:
                raise TypeError(error_message(func, argname, argvalue))
            return argvalue
        return _actual_preprocessor
    return preprocess(**valmap(_expect_dtype, named))

def expect_kinds(**named):
    if False:
        i = 10
        return i + 15
    "\n    Preprocessing decorator that verifies inputs have expected dtype kinds.\n\n    Examples\n    --------\n    >>> from numpy import int64, int32, float32\n    >>> @expect_kinds(x='i')\n    ... def foo(x):\n    ...    return x\n    ...\n    >>> foo(int64(2))\n    2\n    >>> foo(int32(2))\n    2\n    >>> foo(float32(2))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    TypeError: ...foo() expected a numpy object of kind 'i' for argument 'x',\n    but got 'f' instead.\n    "
    for (name, kind) in iteritems(named):
        if not isinstance(kind, (str, tuple)):
            raise TypeError('expect_dtype_kinds() expected a string or tuple of strings for argument {name!r}, but got {kind} instead.'.format(name=name, kind=dtype))

    @preprocess(kinds=call(lambda x: x if isinstance(x, tuple) else (x,)))
    def _expect_kind(kinds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Factory for kind-checking functions that work the @preprocess\n        decorator.\n        '

        def error_message(func, argname, value):
            if False:
                i = 10
                return i + 15
            try:
                value_to_show = value.dtype.kind
            except AttributeError:
                value_to_show = value
            return '{funcname}() expected a numpy object of kind {kinds} for argument {argname!r}, but got {value!r} instead.'.format(funcname=_qualified_name(func), kinds=' or '.join(map(repr, kinds)), argname=argname, value=value_to_show)

        def _actual_preprocessor(func, argname, argvalue):
            if False:
                i = 10
                return i + 15
            if getattrs(argvalue, ('dtype', 'kind'), object()) not in kinds:
                raise TypeError(error_message(func, argname, argvalue))
            return argvalue
        return _actual_preprocessor
    return preprocess(**valmap(_expect_kind, named))

def expect_types(__funcname=_qualified_name, **named):
    if False:
        print('Hello World!')
    "\n    Preprocessing decorator that verifies inputs have expected types.\n\n    Examples\n    --------\n    >>> @expect_types(x=int, y=str)\n    ... def foo(x, y):\n    ...    return x, y\n    ...\n    >>> foo(2, '3')\n    (2, '3')\n    >>> foo(2.0, '3')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    TypeError: ...foo() expected a value of type int for argument 'x',\n    but got float instead.\n\n    Notes\n    -----\n    A special argument, __funcname, can be provided as a string to override the\n    function name shown in error messages.  This is most often used on __init__\n    or __new__ methods to make errors refer to the class name instead of the\n    function name.\n    "
    for (name, type_) in iteritems(named):
        if not isinstance(type_, (type, tuple)):
            raise TypeError("expect_types() expected a type or tuple of types for argument '{name}', but got {type_} instead.".format(name=name, type_=type_))

    def _expect_type(type_):
        if False:
            i = 10
            return i + 15
        _template = "%(funcname)s() expected a value of type {type_or_types} for argument '%(argname)s', but got %(actual)s instead."
        if isinstance(type_, tuple):
            template = _template.format(type_or_types=' or '.join(map(_qualified_name, type_)))
        else:
            template = _template.format(type_or_types=_qualified_name(type_))
        return make_check(exc_type=TypeError, template=template, pred=lambda v: not isinstance(v, type_), actual=compose(_qualified_name, type), funcname=__funcname)
    return preprocess(**valmap(_expect_type, named))

def make_check(exc_type, template, pred, actual, funcname):
    if False:
        print('Hello World!')
    "\n    Factory for making preprocessing functions that check a predicate on the\n    input value.\n\n    Parameters\n    ----------\n    exc_type : Exception\n        The exception type to raise if the predicate fails.\n    template : str\n        A template string to use to create error messages.\n        Should have %-style named template parameters for 'funcname',\n        'argname', and 'actual'.\n    pred : function[object -> bool]\n        A function to call on the argument being preprocessed.  If the\n        predicate returns `True`, we raise an instance of `exc_type`.\n    actual : function[object -> object]\n        A function to call on bad values to produce the value to display in the\n        error message.\n    funcname : str or callable\n        Name to use in error messages, or function to call on decorated\n        functions to produce a name.  Passing an explicit name is useful when\n        creating checks for __init__ or __new__ methods when you want the error\n        to refer to the class name instead of the method name.\n    "
    if isinstance(funcname, str):

        def get_funcname(_):
            if False:
                for i in range(10):
                    print('nop')
            return funcname
    else:
        get_funcname = funcname

    def _check(func, argname, argvalue):
        if False:
            while True:
                i = 10
        if pred(argvalue):
            raise exc_type(template % {'funcname': get_funcname(func), 'argname': argname, 'actual': actual(argvalue)})
        return argvalue
    return _check

def optional(type_):
    if False:
        return 10
    '\n    Helper for use with `expect_types` when an input can be `type_` or `None`.\n\n    Returns an object such that both `None` and instances of `type_` pass\n    checks of the form `isinstance(obj, optional(type_))`.\n\n    Parameters\n    ----------\n    type_ : type\n       Type for which to produce an option.\n\n    Examples\n    --------\n    >>> isinstance({}, optional(dict))\n    True\n    >>> isinstance(None, optional(dict))\n    True\n    >>> isinstance(1, optional(dict))\n    False\n    '
    return (type_, type(None))

def expect_element(__funcname=_qualified_name, **named):
    if False:
        while True:
            i = 10
    "\n    Preprocessing decorator that verifies inputs are elements of some\n    expected collection.\n\n    Examples\n    --------\n    >>> @expect_element(x=('a', 'b'))\n    ... def foo(x):\n    ...    return x.upper()\n    ...\n    >>> foo('a')\n    'A'\n    >>> foo('b')\n    'B'\n    >>> foo('c')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value in ('a', 'b') for argument 'x',\n    but got 'c' instead.\n\n    Notes\n    -----\n    A special argument, __funcname, can be provided as a string to override the\n    function name shown in error messages.  This is most often used on __init__\n    or __new__ methods to make errors refer to the class name instead of the\n    function name.\n\n    This uses the `in` operator (__contains__) to make the containment check.\n    This allows us to use any custom container as long as the object supports\n    the container protocol.\n    "

    def _expect_element(collection):
        if False:
            print('Hello World!')
        if isinstance(collection, (set, frozenset)):
            collection_for_error_message = tuple(sorted(collection))
        else:
            collection_for_error_message = collection
        template = "%(funcname)s() expected a value in {collection} for argument '%(argname)s', but got %(actual)s instead.".format(collection=collection_for_error_message)
        return make_check(ValueError, template, complement(op.contains(collection)), repr, funcname=__funcname)
    return preprocess(**valmap(_expect_element, named))

def expect_bounded(__funcname=_qualified_name, **named):
    if False:
        print('Hello World!')
    "\n    Preprocessing decorator verifying that inputs fall INCLUSIVELY between\n    bounds.\n\n    Bounds should be passed as a pair of ``(min_value, max_value)``.\n\n    ``None`` may be passed as ``min_value`` or ``max_value`` to signify that\n    the input is only bounded above or below.\n\n    Examples\n    --------\n    >>> @expect_bounded(x=(1, 5))\n    ... def foo(x):\n    ...    return x + 1\n    ...\n    >>> foo(1)\n    2\n    >>> foo(5)\n    6\n    >>> foo(6)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value inclusively between 1 and 5 for\n    argument 'x', but got 6 instead.\n\n    >>> @expect_bounded(x=(2, None))\n    ... def foo(x):\n    ...    return x\n    ...\n    >>> foo(100000)\n    100000\n    >>> foo(1)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value greater than or equal to 2 for\n    argument 'x', but got 1 instead.\n\n    >>> @expect_bounded(x=(None, 5))\n    ... def foo(x):\n    ...    return x\n    ...\n    >>> foo(6)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value less than or equal to 5 for\n    argument 'x', but got 6 instead.\n    "

    def _make_bounded_check(bounds):
        if False:
            print('Hello World!')
        (lower, upper) = bounds
        if lower is None:

            def should_fail(value):
                if False:
                    while True:
                        i = 10
                return value > upper
            predicate_descr = 'less than or equal to ' + str(upper)
        elif upper is None:

            def should_fail(value):
                if False:
                    print('Hello World!')
                return value < lower
            predicate_descr = 'greater than or equal to ' + str(lower)
        else:

            def should_fail(value):
                if False:
                    i = 10
                    return i + 15
                return not lower <= value <= upper
            predicate_descr = 'inclusively between %s and %s' % bounds
        template = "%(funcname)s() expected a value {predicate} for argument '%(argname)s', but got %(actual)s instead.".format(predicate=predicate_descr)
        return make_check(exc_type=ValueError, template=template, pred=should_fail, actual=repr, funcname=__funcname)
    return _expect_bounded(_make_bounded_check, __funcname=__funcname, **named)

def expect_strictly_bounded(__funcname=_qualified_name, **named):
    if False:
        return 10
    "\n    Preprocessing decorator verifying that inputs fall EXCLUSIVELY between\n    bounds.\n\n    Bounds should be passed as a pair of ``(min_value, max_value)``.\n\n    ``None`` may be passed as ``min_value`` or ``max_value`` to signify that\n    the input is only bounded above or below.\n\n    Examples\n    --------\n    >>> @expect_strictly_bounded(x=(1, 5))\n    ... def foo(x):\n    ...    return x + 1\n    ...\n    >>> foo(2)\n    3\n    >>> foo(4)\n    5\n    >>> foo(5)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value exclusively between 1 and 5 for\n    argument 'x', but got 5 instead.\n\n    >>> @expect_strictly_bounded(x=(2, None))\n    ... def foo(x):\n    ...    return x\n    ...\n    >>> foo(100000)\n    100000\n    >>> foo(2)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value strictly greater than 2 for\n    argument 'x', but got 2 instead.\n\n    >>> @expect_strictly_bounded(x=(None, 5))\n    ... def foo(x):\n    ...    return x\n    ...\n    >>> foo(5)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a value strictly less than 5 for\n    argument 'x', but got 5 instead.\n    "

    def _make_bounded_check(bounds):
        if False:
            print('Hello World!')
        (lower, upper) = bounds
        if lower is None:

            def should_fail(value):
                if False:
                    return 10
                return value >= upper
            predicate_descr = 'strictly less than ' + str(upper)
        elif upper is None:

            def should_fail(value):
                if False:
                    print('Hello World!')
                return value <= lower
            predicate_descr = 'strictly greater than ' + str(lower)
        else:

            def should_fail(value):
                if False:
                    while True:
                        i = 10
                return not lower < value < upper
            predicate_descr = 'exclusively between %s and %s' % bounds
        template = "%(funcname)s() expected a value {predicate} for argument '%(argname)s', but got %(actual)s instead.".format(predicate=predicate_descr)
        return make_check(exc_type=ValueError, template=template, pred=should_fail, actual=repr, funcname=__funcname)
    return _expect_bounded(_make_bounded_check, __funcname=__funcname, **named)

def _expect_bounded(make_bounded_check, __funcname, **named):
    if False:
        while True:
            i = 10

    def valid_bounds(t):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(t, tuple) and len(t) == 2 and (t != (None, None))
    for (name, bounds) in iteritems(named):
        if not valid_bounds(bounds):
            raise TypeError("expect_bounded() expected a tuple of bounds for argument '{name}', but got {bounds} instead.".format(name=name, bounds=bounds))
    return preprocess(**valmap(make_bounded_check, named))

def expect_dimensions(__funcname=_qualified_name, **dimensions):
    if False:
        print('Hello World!')
    "\n    Preprocessing decorator that verifies inputs are numpy arrays with a\n    specific dimensionality.\n\n    Examples\n    --------\n    >>> from numpy import array\n    >>> @expect_dimensions(x=1, y=2)\n    ... def foo(x, y):\n    ...    return x[0] + y[0, 0]\n    ...\n    >>> foo(array([1, 1]), array([[1, 1], [2, 2]]))\n    2\n    >>> foo(array([1, 1]), array([1, 1]))  # doctest: +NORMALIZE_WHITESPACE\n    ...                                    # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n       ...\n    ValueError: ...foo() expected a 2-D array for argument 'y',\n    but got a 1-D array instead.\n    "
    if isinstance(__funcname, str):

        def get_funcname(_):
            if False:
                for i in range(10):
                    print('nop')
            return __funcname
    else:
        get_funcname = __funcname

    def _expect_dimension(expected_ndim):
        if False:
            return 10

        def _check(func, argname, argvalue):
            if False:
                return 10
            actual_ndim = argvalue.ndim
            if actual_ndim != expected_ndim:
                if actual_ndim == 0:
                    actual_repr = 'scalar'
                else:
                    actual_repr = '%d-D array' % actual_ndim
                raise ValueError('{func}() expected a {expected:d}-D array for argument {argname!r}, but got a {actual} instead.'.format(func=get_funcname(func), expected=expected_ndim, argname=argname, actual=actual_repr))
            return argvalue
        return _check
    return preprocess(**valmap(_expect_dimension, dimensions))

def coerce(from_, to, **to_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    A preprocessing decorator that coerces inputs of a given type by passing\n    them to a callable.\n\n    Parameters\n    ----------\n    from : type or tuple or types\n        Inputs types on which to call ``to``.\n    to : function\n        Coercion function to call on inputs.\n    **to_kwargs\n        Additional keywords to forward to every call to ``to``.\n\n    Examples\n    --------\n    >>> @preprocess(x=coerce(float, int), y=coerce(float, int))\n    ... def floordiff(x, y):\n    ...     return x - y\n    ...\n    >>> floordiff(3.2, 2.5)\n    1\n\n    >>> @preprocess(x=coerce(str, int, base=2), y=coerce(str, int, base=2))\n    ... def add_binary_strings(x, y):\n    ...     return bin(x + y)[2:]\n    ...\n    >>> add_binary_strings('101', '001')\n    '110'\n    "

    def preprocessor(func, argname, arg):
        if False:
            i = 10
            return i + 15
        if isinstance(arg, from_):
            return to(arg, **to_kwargs)
        return arg
    return preprocessor

def coerce_types(**kwargs):
    if False:
        print('Hello World!')
    "\n    Preprocessing decorator that applies type coercions.\n\n    Parameters\n    ----------\n    **kwargs : dict[str -> (type, callable)]\n         Keyword arguments mapping function parameter names to pairs of\n         (from_type, to_type).\n\n    Examples\n    --------\n    >>> @coerce_types(x=(float, int), y=(int, str))\n    ... def func(x, y):\n    ...     return (x, y)\n    ...\n    >>> func(1.0, 3)\n    (1, '3')\n    "

    def _coerce(types):
        if False:
            i = 10
            return i + 15
        return coerce(*types)
    return preprocess(**valmap(_coerce, kwargs))

class error_keywords(object):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.messages = kwargs

    def __call__(self, func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def assert_keywords_and_call(*args, **kwargs):
            if False:
                return 10
            for (field, message) in iteritems(self.messages):
                if field in kwargs:
                    raise TypeError(message)
            return func(*args, **kwargs)
        return assert_keywords_and_call
coerce_string = partial(coerce, string_types)

def validate_keys(dict_, expected, funcname):
    if False:
        for i in range(10):
            print('nop')
    'Validate that a dictionary has an expected set of keys.\n    '
    expected = set(expected)
    received = set(dict_)
    missing = expected - received
    if missing:
        raise ValueError('Missing keys in {}:\nExpected Keys: {}\nReceived Keys: {}'.format(funcname, sorted(expected), sorted(received)))
    unexpected = received - expected
    if unexpected:
        raise ValueError('Unexpected keys in {}:\nExpected Keys: {}\nReceived Keys: {}'.format(funcname, sorted(expected), sorted(received)))