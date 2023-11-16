"""Array printing function

$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $

"""
__all__ = ['array2string', 'array_str', 'array_repr', 'set_printoptions', 'get_printoptions', 'printoptions', 'format_float_positional', 'format_float_scientific']
__docformat__ = 'restructuredtext'
import functools
import numbers
import sys
try:
    from _thread import get_ident
except ImportError:
    from _dummy_thread import get_ident
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import array, dragon4_positional, dragon4_scientific, datetime_as_string, datetime_data, ndarray, set_legacy_print_mode
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import longlong, intc, int_, float64, complex128, bool_, flexible
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
_format_options = {'edgeitems': 3, 'threshold': 1000, 'floatmode': 'maxprec', 'precision': 8, 'suppress': False, 'linewidth': 75, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-', 'formatter': None, 'legacy': sys.maxsize}

def _make_options_dict(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, sign=None, formatter=None, floatmode=None, legacy=None):
    if False:
        i = 10
        return i + 15
    '\n    Make a dictionary out of the non-None arguments, plus conversion of\n    *legacy* and sanity checks.\n    '
    options = {k: v for (k, v) in list(locals().items()) if v is not None}
    if suppress is not None:
        options['suppress'] = bool(suppress)
    modes = ['fixed', 'unique', 'maxprec', 'maxprec_equal']
    if floatmode not in modes + [None]:
        raise ValueError('floatmode option must be one of ' + ', '.join(('"{}"'.format(m) for m in modes)))
    if sign not in [None, '-', '+', ' ']:
        raise ValueError("sign option must be one of ' ', '+', or '-'")
    if legacy == False:
        options['legacy'] = sys.maxsize
    elif legacy == '1.13':
        options['legacy'] = 113
    elif legacy == '1.21':
        options['legacy'] = 121
    elif legacy == '1.25':
        options['legacy'] = 125
    elif legacy is None:
        pass
    else:
        warnings.warn("legacy printing option can currently only be '1.13', '1.21', '1.25', or `False`", stacklevel=3)
    if threshold is not None:
        if not isinstance(threshold, numbers.Number):
            raise TypeError('threshold must be numeric')
        if np.isnan(threshold):
            raise ValueError('threshold must be non-NAN, try sys.maxsize for untruncated representation')
    if precision is not None:
        try:
            options['precision'] = operator.index(precision)
        except TypeError as e:
            raise TypeError('precision must be an integer') from e
    return options

@set_module('numpy')
def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None, *, legacy=None):
    if False:
        while True:
            i = 10
    "\n    Set printing options.\n\n    These options determine the way floating point numbers, arrays and\n    other NumPy objects are displayed.\n\n    Parameters\n    ----------\n    precision : int or None, optional\n        Number of digits of precision for floating point output (default 8).\n        May be None if `floatmode` is not `fixed`, to print as many digits as\n        necessary to uniquely specify the value.\n    threshold : int, optional\n        Total number of array elements which trigger summarization\n        rather than full repr (default 1000).\n        To always use the full repr without summarization, pass `sys.maxsize`.\n    edgeitems : int, optional\n        Number of array items in summary at beginning and end of\n        each dimension (default 3).\n    linewidth : int, optional\n        The number of characters per line for the purpose of inserting\n        line breaks (default 75).\n    suppress : bool, optional\n        If True, always print floating point numbers using fixed point\n        notation, in which case numbers equal to zero in the current precision\n        will print as zero.  If False, then scientific notation is used when\n        absolute value of the smallest number is < 1e-4 or the ratio of the\n        maximum absolute value to the minimum is > 1e3. The default is False.\n    nanstr : str, optional\n        String representation of floating point not-a-number (default nan).\n    infstr : str, optional\n        String representation of floating point infinity (default inf).\n    sign : string, either '-', '+', or ' ', optional\n        Controls printing of the sign of floating-point types. If '+', always\n        print the sign of positive values. If ' ', always prints a space\n        (whitespace character) in the sign position of positive values.  If\n        '-', omit the sign character of positive values. (default '-')\n\n        .. versionchanged:: 2.0\n             The sign parameter can now be an integer type, previously\n             types were floating-point types.\n\n    formatter : dict of callables, optional\n        If not None, the keys should indicate the type(s) that the respective\n        formatting function applies to.  Callables should return a string.\n        Types that are not specified (by their corresponding keys) are handled\n        by the default formatters.  Individual types for which a formatter\n        can be set are:\n\n        - 'bool'\n        - 'int'\n        - 'timedelta' : a `numpy.timedelta64`\n        - 'datetime' : a `numpy.datetime64`\n        - 'float'\n        - 'longfloat' : 128-bit floats\n        - 'complexfloat'\n        - 'longcomplexfloat' : composed of two 128-bit floats\n        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`\n        - 'object' : `np.object_` arrays\n\n        Other keys that can be used to set a group of types at once are:\n\n        - 'all' : sets all types\n        - 'int_kind' : sets 'int'\n        - 'float_kind' : sets 'float' and 'longfloat'\n        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'\n        - 'str_kind' : sets 'numpystr'\n    floatmode : str, optional\n        Controls the interpretation of the `precision` option for\n        floating-point types. Can take the following values\n        (default maxprec_equal):\n\n        * 'fixed': Always print exactly `precision` fractional digits,\n                even if this would print more or fewer digits than\n                necessary to specify the value uniquely.\n        * 'unique': Print the minimum number of fractional digits necessary\n                to represent each value uniquely. Different elements may\n                have a different number of digits. The value of the\n                `precision` option is ignored.\n        * 'maxprec': Print at most `precision` fractional digits, but if\n                an element can be uniquely represented with fewer digits\n                only print it with that many.\n        * 'maxprec_equal': Print at most `precision` fractional digits,\n                but if every element in the array can be uniquely\n                represented with an equal number of fewer digits, use that\n                many digits for all elements.\n    legacy : string or `False`, optional\n        If set to the string `'1.13'` enables 1.13 legacy printing mode. This\n        approximates numpy 1.13 print output by including a space in the sign\n        position of floats and different behavior for 0d arrays. This also\n        enables 1.21 legacy printing mode (described below).\n\n        If set to the string `'1.21'` enables 1.21 legacy printing mode. This\n        approximates numpy 1.21 print output of complex structured dtypes\n        by not inserting spaces after commas that separate fields and after\n        colons.\n\n        If set to `False`, disables legacy mode.\n\n        Unrecognized strings will be ignored with a warning for forward\n        compatibility.\n\n        .. versionadded:: 1.14.0\n        .. versionchanged:: 1.22.0\n\n    See Also\n    --------\n    get_printoptions, printoptions, array2string\n\n    Notes\n    -----\n    `formatter` is always reset with a call to `set_printoptions`.\n\n    Use `printoptions` as a context manager to set the values temporarily.\n\n    Examples\n    --------\n    Floating point precision can be set:\n\n    >>> np.set_printoptions(precision=4)\n    >>> np.array([1.123456789])\n    [1.1235]\n\n    Long arrays can be summarised:\n\n    >>> np.set_printoptions(threshold=5)\n    >>> np.arange(10)\n    array([0, 1, 2, ..., 7, 8, 9])\n\n    Small results can be suppressed:\n\n    >>> eps = np.finfo(float).eps\n    >>> x = np.arange(4.)\n    >>> x**2 - (x + eps)**2\n    array([-4.9304e-32, -4.4409e-16,  0.0000e+00,  0.0000e+00])\n    >>> np.set_printoptions(suppress=True)\n    >>> x**2 - (x + eps)**2\n    array([-0., -0.,  0.,  0.])\n\n    A custom formatter can be used to display array elements as desired:\n\n    >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})\n    >>> x = np.arange(3)\n    >>> x\n    array([int: 0, int: -1, int: -2])\n    >>> np.set_printoptions()  # formatter gets reset\n    >>> x\n    array([0, 1, 2])\n\n    To put back the default options, you can use:\n\n    >>> np.set_printoptions(edgeitems=3, infstr='inf',\n    ... linewidth=75, nanstr='nan', precision=8,\n    ... suppress=False, threshold=1000, formatter=None)\n\n    Also to temporarily override options, use `printoptions`\n    as a context manager:\n\n    >>> with np.printoptions(precision=2, suppress=True, threshold=5):\n    ...     np.linspace(0, 10, 10)\n    array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ])\n\n    "
    opt = _make_options_dict(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, sign, formatter, floatmode, legacy)
    opt['formatter'] = formatter
    _format_options.update(opt)
    if _format_options['legacy'] == 113:
        set_legacy_print_mode(113)
        _format_options['sign'] = '-'
    elif _format_options['legacy'] == 121:
        set_legacy_print_mode(121)
    elif _format_options['legacy'] == 125:
        set_legacy_print_mode(125)
    elif _format_options['legacy'] == sys.maxsize:
        set_legacy_print_mode(0)

@set_module('numpy')
def get_printoptions():
    if False:
        return 10
    '\n    Return the current print options.\n\n    Returns\n    -------\n    print_opts : dict\n        Dictionary of current print options with keys\n\n        - precision : int\n        - threshold : int\n        - edgeitems : int\n        - linewidth : int\n        - suppress : bool\n        - nanstr : str\n        - infstr : str\n        - formatter : dict of callables\n        - sign : str\n\n        For a full description of these options, see `set_printoptions`.\n\n    See Also\n    --------\n    set_printoptions, printoptions\n\n    '
    opts = _format_options.copy()
    opts['legacy'] = {113: '1.13', 121: '1.21', 125: '1.25', sys.maxsize: False}[opts['legacy']]
    return opts

def _get_legacy_print_mode():
    if False:
        print('Hello World!')
    'Return the legacy print mode as an int.'
    return _format_options['legacy']

@set_module('numpy')
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Context manager for setting print options.\n\n    Set print options for the scope of the `with` block, and restore the old\n    options at the end. See `set_printoptions` for the full description of\n    available options.\n\n    Examples\n    --------\n\n    >>> from numpy.testing import assert_equal\n    >>> with np.printoptions(precision=2):\n    ...     np.array([2.0]) / 3\n    array([0.67])\n\n    The `as`-clause of the `with`-statement gives the current print options:\n\n    >>> with np.printoptions(precision=2) as opts:\n    ...      assert_equal(opts, np.get_printoptions())\n\n    See Also\n    --------\n    set_printoptions, get_printoptions\n\n    '
    opts = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield np.get_printoptions()
    finally:
        np.set_printoptions(**opts)

def _leading_trailing(a, edgeitems, index=()):
    if False:
        for i in range(10):
            print('nop')
    '\n    Keep only the N-D corners (leading and trailing edges) of an array.\n\n    Should be passed a base-class ndarray, since it makes no guarantees about\n    preserving subclasses.\n    '
    axis = len(index)
    if axis == a.ndim:
        return a[index]
    if a.shape[axis] > 2 * edgeitems:
        return concatenate((_leading_trailing(a, edgeitems, index + np.index_exp[:edgeitems]), _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:])), axis=axis)
    else:
        return _leading_trailing(a, edgeitems, index + np.index_exp[:])

def _object_format(o):
    if False:
        while True:
            i = 10
    ' Object arrays containing lists should be printed unambiguously '
    if type(o) is list:
        fmt = 'list({!r})'
    else:
        fmt = '{!r}'
    return fmt.format(o)

def repr_format(x):
    if False:
        return 10
    if isinstance(x, (np.str_, np.bytes_)):
        return repr(x.item())
    return repr(x)

def str_format(x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, (np.str_, np.bytes_)):
        return str(x.item())
    return str(x)

def _get_formatdict(data, *, precision, floatmode, suppress, sign, legacy, formatter, **kwargs):
    if False:
        i = 10
        return i + 15
    formatdict = {'bool': lambda : BoolFormat(data), 'int': lambda : IntegerFormat(data, sign), 'float': lambda : FloatingFormat(data, precision, floatmode, suppress, sign, legacy=legacy), 'longfloat': lambda : FloatingFormat(data, precision, floatmode, suppress, sign, legacy=legacy), 'complexfloat': lambda : ComplexFloatingFormat(data, precision, floatmode, suppress, sign, legacy=legacy), 'longcomplexfloat': lambda : ComplexFloatingFormat(data, precision, floatmode, suppress, sign, legacy=legacy), 'datetime': lambda : DatetimeFormat(data, legacy=legacy), 'timedelta': lambda : TimedeltaFormat(data), 'object': lambda : _object_format, 'void': lambda : str_format, 'numpystr': lambda : repr_format}

    def indirect(x):
        if False:
            while True:
                i = 10
        return lambda : x
    if formatter is not None:
        fkeys = [k for k in formatter.keys() if formatter[k] is not None]
        if 'all' in fkeys:
            for key in formatdict.keys():
                formatdict[key] = indirect(formatter['all'])
        if 'int_kind' in fkeys:
            for key in ['int']:
                formatdict[key] = indirect(formatter['int_kind'])
        if 'float_kind' in fkeys:
            for key in ['float', 'longfloat']:
                formatdict[key] = indirect(formatter['float_kind'])
        if 'complex_kind' in fkeys:
            for key in ['complexfloat', 'longcomplexfloat']:
                formatdict[key] = indirect(formatter['complex_kind'])
        if 'str_kind' in fkeys:
            formatdict['numpystr'] = indirect(formatter['str_kind'])
        for key in formatdict.keys():
            if key in fkeys:
                formatdict[key] = indirect(formatter[key])
    return formatdict

def _get_format_function(data, **options):
    if False:
        while True:
            i = 10
    '\n    find the right formatting function for the dtype_\n    '
    dtype_ = data.dtype
    dtypeobj = dtype_.type
    formatdict = _get_formatdict(data, **options)
    if dtypeobj is None:
        return formatdict['numpystr']()
    elif issubclass(dtypeobj, _nt.bool_):
        return formatdict['bool']()
    elif issubclass(dtypeobj, _nt.integer):
        if issubclass(dtypeobj, _nt.timedelta64):
            return formatdict['timedelta']()
        else:
            return formatdict['int']()
    elif issubclass(dtypeobj, _nt.floating):
        if issubclass(dtypeobj, _nt.longdouble):
            return formatdict['longfloat']()
        else:
            return formatdict['float']()
    elif issubclass(dtypeobj, _nt.complexfloating):
        if issubclass(dtypeobj, _nt.clongdouble):
            return formatdict['longcomplexfloat']()
        else:
            return formatdict['complexfloat']()
    elif issubclass(dtypeobj, (_nt.str_, _nt.bytes_)):
        return formatdict['numpystr']()
    elif issubclass(dtypeobj, _nt.datetime64):
        return formatdict['datetime']()
    elif issubclass(dtypeobj, _nt.object_):
        return formatdict['object']()
    elif issubclass(dtypeobj, _nt.void):
        if dtype_.names is not None:
            return StructuredVoidFormat.from_data(data, **options)
        else:
            return formatdict['void']()
    else:
        return formatdict['numpystr']()

def _recursive_guard(fillvalue='...'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs\n\n    Decorates a function such that if it calls itself with the same first\n    argument, it returns `fillvalue` instead of recursing.\n\n    Largely copied from reprlib.recursive_repr\n    '

    def decorating_function(f):
        if False:
            for i in range(10):
                print('nop')
        repr_running = set()

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            key = (id(self), get_ident())
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                return f(self, *args, **kwargs)
            finally:
                repr_running.discard(key)
        return wrapper
    return decorating_function

@_recursive_guard()
def _array2string(a, options, separator=' ', prefix=''):
    if False:
        while True:
            i = 10
    data = asarray(a)
    if a.shape == ():
        a = data
    if a.size > options['threshold']:
        summary_insert = '...'
        data = _leading_trailing(data, options['edgeitems'])
    else:
        summary_insert = ''
    format_function = _get_format_function(data, **options)
    next_line_prefix = ' '
    next_line_prefix += ' ' * len(prefix)
    lst = _formatArray(a, format_function, options['linewidth'], next_line_prefix, separator, options['edgeitems'], summary_insert, options['legacy'])
    return lst

def _array2string_dispatcher(a, max_line_width=None, precision=None, suppress_small=None, separator=None, prefix=None, style=None, formatter=None, threshold=None, edgeitems=None, sign=None, floatmode=None, suffix=None, *, legacy=None):
    if False:
        while True:
            i = 10
    return (a,)

@array_function_dispatch(_array2string_dispatcher, module='numpy')
def array2string(a, max_line_width=None, precision=None, suppress_small=None, separator=' ', prefix='', style=np._NoValue, formatter=None, threshold=None, edgeitems=None, sign=None, floatmode=None, suffix='', *, legacy=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a string representation of an array.\n\n    Parameters\n    ----------\n    a : ndarray\n        Input array.\n    max_line_width : int, optional\n        Inserts newlines if text is longer than `max_line_width`.\n        Defaults to ``numpy.get_printoptions()[\'linewidth\']``.\n    precision : int or None, optional\n        Floating point precision.\n        Defaults to ``numpy.get_printoptions()[\'precision\']``.\n    suppress_small : bool, optional\n        Represent numbers "very close" to zero as zero; default is False.\n        Very close is defined by precision: if the precision is 8, e.g.,\n        numbers smaller (in absolute value) than 5e-9 are represented as\n        zero.\n        Defaults to ``numpy.get_printoptions()[\'suppress\']``.\n    separator : str, optional\n        Inserted between elements.\n    prefix : str, optional\n    suffix : str, optional\n        The length of the prefix and suffix strings are used to respectively\n        align and wrap the output. An array is typically printed as::\n\n          prefix + array2string(a) + suffix\n\n        The output is left-padded by the length of the prefix string, and\n        wrapping is forced at the column ``max_line_width - len(suffix)``.\n        It should be noted that the content of prefix and suffix strings are\n        not included in the output.\n    style : _NoValue, optional\n        Has no effect, do not use.\n\n        .. deprecated:: 1.14.0\n    formatter : dict of callables, optional\n        If not None, the keys should indicate the type(s) that the respective\n        formatting function applies to.  Callables should return a string.\n        Types that are not specified (by their corresponding keys) are handled\n        by the default formatters.  Individual types for which a formatter\n        can be set are:\n\n        - \'bool\'\n        - \'int\'\n        - \'timedelta\' : a `numpy.timedelta64`\n        - \'datetime\' : a `numpy.datetime64`\n        - \'float\'\n        - \'longfloat\' : 128-bit floats\n        - \'complexfloat\'\n        - \'longcomplexfloat\' : composed of two 128-bit floats\n        - \'void\' : type `numpy.void`\n        - \'numpystr\' : types `numpy.bytes_` and `numpy.str_`\n\n        Other keys that can be used to set a group of types at once are:\n\n        - \'all\' : sets all types\n        - \'int_kind\' : sets \'int\'\n        - \'float_kind\' : sets \'float\' and \'longfloat\'\n        - \'complex_kind\' : sets \'complexfloat\' and \'longcomplexfloat\'\n        - \'str_kind\' : sets \'numpystr\'\n    threshold : int, optional\n        Total number of array elements which trigger summarization\n        rather than full repr.\n        Defaults to ``numpy.get_printoptions()[\'threshold\']``.\n    edgeitems : int, optional\n        Number of array items in summary at beginning and end of\n        each dimension.\n        Defaults to ``numpy.get_printoptions()[\'edgeitems\']``.\n    sign : string, either \'-\', \'+\', or \' \', optional\n        Controls printing of the sign of floating-point types. If \'+\', always\n        print the sign of positive values. If \' \', always prints a space\n        (whitespace character) in the sign position of positive values.  If\n        \'-\', omit the sign character of positive values.\n        Defaults to ``numpy.get_printoptions()[\'sign\']``.\n\n        .. versionchanged:: 2.0\n             The sign parameter can now be an integer type, previously\n             types were floating-point types.\n\n    floatmode : str, optional\n        Controls the interpretation of the `precision` option for\n        floating-point types.\n        Defaults to ``numpy.get_printoptions()[\'floatmode\']``.\n        Can take the following values:\n\n        - \'fixed\': Always print exactly `precision` fractional digits,\n          even if this would print more or fewer digits than\n          necessary to specify the value uniquely.\n        - \'unique\': Print the minimum number of fractional digits necessary\n          to represent each value uniquely. Different elements may\n          have a different number of digits.  The value of the\n          `precision` option is ignored.\n        - \'maxprec\': Print at most `precision` fractional digits, but if\n          an element can be uniquely represented with fewer digits\n          only print it with that many.\n        - \'maxprec_equal\': Print at most `precision` fractional digits,\n          but if every element in the array can be uniquely\n          represented with an equal number of fewer digits, use that\n          many digits for all elements.\n    legacy : string or `False`, optional\n        If set to the string `\'1.13\'` enables 1.13 legacy printing mode. This\n        approximates numpy 1.13 print output by including a space in the sign\n        position of floats and different behavior for 0d arrays. If set to\n        `False`, disables legacy mode. Unrecognized strings will be ignored\n        with a warning for forward compatibility.\n\n        .. versionadded:: 1.14.0\n\n    Returns\n    -------\n    array_str : str\n        String representation of the array.\n\n    Raises\n    ------\n    TypeError\n        if a callable in `formatter` does not return a string.\n\n    See Also\n    --------\n    array_str, array_repr, set_printoptions, get_printoptions\n\n    Notes\n    -----\n    If a formatter is specified for a certain type, the `precision` keyword is\n    ignored for that type.\n\n    This is a very flexible function; `array_repr` and `array_str` are using\n    `array2string` internally so keywords with the same name should work\n    identically in all three functions.\n\n    Examples\n    --------\n    >>> x = np.array([1e-16,1,2,3])\n    >>> np.array2string(x, precision=2, separator=\',\',\n    ...                       suppress_small=True)\n    \'[0.,1.,2.,3.]\'\n\n    >>> x  = np.arange(3.)\n    >>> np.array2string(x, formatter={\'float_kind\':lambda x: "%.2f" % x})\n    \'[0.00 1.00 2.00]\'\n\n    >>> x  = np.arange(3)\n    >>> np.array2string(x, formatter={\'int\':lambda x: hex(x)})\n    \'[0x0 0x1 0x2]\'\n\n    '
    overrides = _make_options_dict(precision, threshold, edgeitems, max_line_width, suppress_small, None, None, sign, formatter, floatmode, legacy)
    options = _format_options.copy()
    options.update(overrides)
    if options['legacy'] <= 113:
        if style is np._NoValue:
            style = repr
        if a.shape == () and a.dtype.names is None:
            return style(a.item())
    elif style is not np._NoValue:
        warnings.warn("'style' argument is deprecated and no longer functional except in 1.13 'legacy' mode", DeprecationWarning, stacklevel=2)
    if options['legacy'] > 113:
        options['linewidth'] -= len(suffix)
    if a.size == 0:
        return '[]'
    return _array2string(a, options, separator, prefix)

def _extendLine(s, line, word, line_width, next_line_prefix, legacy):
    if False:
        while True:
            i = 10
    needs_wrap = len(line) + len(word) > line_width
    if legacy > 113:
        if len(line) <= len(next_line_prefix):
            needs_wrap = False
    if needs_wrap:
        s += line.rstrip() + '\n'
        line = next_line_prefix
    line += word
    return (s, line)

def _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy):
    if False:
        i = 10
        return i + 15
    '\n    Extends line with nicely formatted (possibly multi-line) string ``word``.\n    '
    words = word.splitlines()
    if len(words) == 1 or legacy <= 113:
        return _extendLine(s, line, word, line_width, next_line_prefix, legacy)
    max_word_length = max((len(word) for word in words))
    if len(line) + max_word_length > line_width and len(line) > len(next_line_prefix):
        s += line.rstrip() + '\n'
        line = next_line_prefix + words[0]
        indent = next_line_prefix
    else:
        indent = len(line) * ' '
        line += words[0]
    for word in words[1:]:
        s += line.rstrip() + '\n'
        line = indent + word
    suffix_length = max_word_length - len(words[-1])
    line += suffix_length * ' '
    return (s, line)

def _formatArray(a, format_function, line_width, next_line_prefix, separator, edge_items, summary_insert, legacy):
    if False:
        for i in range(10):
            print('nop')
    'formatArray is designed for two modes of operation:\n\n    1. Full output\n\n    2. Summarized output\n\n    '

    def recurser(index, hanging_indent, curr_width):
        if False:
            return 10
        "\n        By using this local function, we don't need to recurse with all the\n        arguments. Since this function is not created recursively, the cost is\n        not significant\n        "
        axis = len(index)
        axes_left = a.ndim - axis
        if axes_left == 0:
            return format_function(a[index])
        next_hanging_indent = hanging_indent + ' '
        if legacy <= 113:
            next_width = curr_width
        else:
            next_width = curr_width - len(']')
        a_len = a.shape[axis]
        show_summary = summary_insert and 2 * edge_items < a_len
        if show_summary:
            leading_items = edge_items
            trailing_items = edge_items
        else:
            leading_items = 0
            trailing_items = a_len
        s = ''
        if axes_left == 1:
            if legacy <= 113:
                elem_width = curr_width - len(separator.rstrip())
            else:
                elem_width = curr_width - max(len(separator.rstrip()), len(']'))
            line = hanging_indent
            for i in range(leading_items):
                word = recurser(index + (i,), next_hanging_indent, next_width)
                (s, line) = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
                line += separator
            if show_summary:
                (s, line) = _extendLine(s, line, summary_insert, elem_width, hanging_indent, legacy)
                if legacy <= 113:
                    line += ', '
                else:
                    line += separator
            for i in range(trailing_items, 1, -1):
                word = recurser(index + (-i,), next_hanging_indent, next_width)
                (s, line) = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
                line += separator
            if legacy <= 113:
                elem_width = curr_width
            word = recurser(index + (-1,), next_hanging_indent, next_width)
            (s, line) = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
            s += line
        else:
            s = ''
            line_sep = separator.rstrip() + '\n' * (axes_left - 1)
            for i in range(leading_items):
                nested = recurser(index + (i,), next_hanging_indent, next_width)
                s += hanging_indent + nested + line_sep
            if show_summary:
                if legacy <= 113:
                    s += hanging_indent + summary_insert + ', \n'
                else:
                    s += hanging_indent + summary_insert + line_sep
            for i in range(trailing_items, 1, -1):
                nested = recurser(index + (-i,), next_hanging_indent, next_width)
                s += hanging_indent + nested + line_sep
            nested = recurser(index + (-1,), next_hanging_indent, next_width)
            s += hanging_indent + nested
        s = '[' + s[len(hanging_indent):] + ']'
        return s
    try:
        return recurser(index=(), hanging_indent=next_line_prefix, curr_width=line_width)
    finally:
        recurser = None

def _none_or_positive_arg(x, name):
    if False:
        for i in range(10):
            print('nop')
    if x is None:
        return -1
    if x < 0:
        raise ValueError('{} must be >= 0'.format(name))
    return x

class FloatingFormat:
    """ Formatter for subtypes of np.floating """

    def __init__(self, data, precision, floatmode, suppress_small, sign=False, *, legacy=None):
        if False:
            while True:
                i = 10
        if isinstance(sign, bool):
            sign = '+' if sign else '-'
        self._legacy = legacy
        if self._legacy <= 113:
            if data.shape != () and sign == '-':
                sign = ' '
        self.floatmode = floatmode
        if floatmode == 'unique':
            self.precision = None
        else:
            self.precision = precision
        self.precision = _none_or_positive_arg(self.precision, 'precision')
        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False
        self.fillFormat(data)

    def fillFormat(self, data):
        if False:
            return 10
        finite_vals = data[isfinite(data)]
        abs_non_zero = absolute(finite_vals[finite_vals != 0])
        if len(abs_non_zero) != 0:
            max_val = np.max(abs_non_zero)
            min_val = np.min(abs_non_zero)
            with errstate(over='ignore'):
                if max_val >= 100000000.0 or (not self.suppress_small and (min_val < 0.0001 or max_val / min_val > 1000.0)):
                    self.exp_format = True
        if len(finite_vals) == 0:
            self.pad_left = 0
            self.pad_right = 0
            self.trim = '.'
            self.exp_size = -1
            self.unique = True
            self.min_digits = None
        elif self.exp_format:
            (trim, unique) = ('.', True)
            if self.floatmode == 'fixed' or self._legacy <= 113:
                (trim, unique) = ('k', False)
            strs = (dragon4_scientific(x, precision=self.precision, unique=unique, trim=trim, sign=self.sign == '+') for x in finite_vals)
            (frac_strs, _, exp_strs) = zip(*(s.partition('e') for s in strs))
            (int_part, frac_part) = zip(*(s.split('.') for s in frac_strs))
            self.exp_size = max((len(s) for s in exp_strs)) - 1
            self.trim = 'k'
            self.precision = max((len(s) for s in frac_part))
            self.min_digits = self.precision
            self.unique = unique
            if self._legacy <= 113:
                self.pad_left = 3
            else:
                self.pad_left = max((len(s) for s in int_part))
            self.pad_right = self.exp_size + 2 + self.precision
        else:
            (trim, unique) = ('.', True)
            if self.floatmode == 'fixed':
                (trim, unique) = ('k', False)
            strs = (dragon4_positional(x, precision=self.precision, fractional=True, unique=unique, trim=trim, sign=self.sign == '+') for x in finite_vals)
            (int_part, frac_part) = zip(*(s.split('.') for s in strs))
            if self._legacy <= 113:
                self.pad_left = 1 + max((len(s.lstrip('-+')) for s in int_part))
            else:
                self.pad_left = max((len(s) for s in int_part))
            self.pad_right = max((len(s) for s in frac_part))
            self.exp_size = -1
            self.unique = unique
            if self.floatmode in ['fixed', 'maxprec_equal']:
                self.precision = self.min_digits = self.pad_right
                self.trim = 'k'
            else:
                self.trim = '.'
                self.min_digits = 0
        if self._legacy > 113:
            if self.sign == ' ' and (not any(np.signbit(finite_vals))):
                self.pad_left += 1
        if data.size != finite_vals.size:
            neginf = self.sign != '-' or any(data[isinf(data)] < 0)
            nanlen = len(_format_options['nanstr'])
            inflen = len(_format_options['infstr']) + neginf
            offset = self.pad_right + 1
            self.pad_left = max(self.pad_left, nanlen - offset, inflen - offset)

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not np.isfinite(x):
            with errstate(invalid='ignore'):
                if np.isnan(x):
                    sign = '+' if self.sign == '+' else ''
                    ret = sign + _format_options['nanstr']
                else:
                    sign = '-' if x < 0 else '+' if self.sign == '+' else ''
                    ret = sign + _format_options['infstr']
                return ' ' * (self.pad_left + self.pad_right + 1 - len(ret)) + ret
        if self.exp_format:
            return dragon4_scientific(x, precision=self.precision, min_digits=self.min_digits, unique=self.unique, trim=self.trim, sign=self.sign == '+', pad_left=self.pad_left, exp_digits=self.exp_size)
        else:
            return dragon4_positional(x, precision=self.precision, min_digits=self.min_digits, unique=self.unique, fractional=True, trim=self.trim, sign=self.sign == '+', pad_left=self.pad_left, pad_right=self.pad_right)

@set_module('numpy')
def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, pad_left=None, exp_digits=None, min_digits=None):
    if False:
        print('Hello World!')
    '\n    Format a floating-point scalar as a decimal string in scientific notation.\n\n    Provides control over rounding, trimming and padding. Uses and assumes\n    IEEE unbiased rounding. Uses the "Dragon4" algorithm.\n\n    Parameters\n    ----------\n    x : python float or numpy floating scalar\n        Value to format.\n    precision : non-negative integer or None, optional\n        Maximum number of digits to print. May be None if `unique` is\n        `True`, but must be an integer if unique is `False`.\n    unique : boolean, optional\n        If `True`, use a digit-generation strategy which gives the shortest\n        representation which uniquely identifies the floating-point number from\n        other values of the same type, by judicious rounding. If `precision`\n        is given fewer digits than necessary can be printed. If `min_digits`\n        is given more can be printed, in which cases the last digit is rounded\n        with unbiased rounding.\n        If `False`, digits are generated as if printing an infinite-precision\n        value and stopping after `precision` digits, rounding the remaining\n        value with unbiased rounding\n    trim : one of \'k\', \'.\', \'0\', \'-\', optional\n        Controls post-processing trimming of trailing digits, as follows:\n\n        * \'k\' : keep trailing zeros, keep decimal point (no trimming)\n        * \'.\' : trim all trailing zeros, leave decimal point\n        * \'0\' : trim all but the zero before the decimal point. Insert the\n          zero if it is missing.\n        * \'-\' : trim trailing zeros and any trailing decimal point\n    sign : boolean, optional\n        Whether to show the sign for positive values.\n    pad_left : non-negative integer, optional\n        Pad the left side of the string with whitespace until at least that\n        many characters are to the left of the decimal point.\n    exp_digits : non-negative integer, optional\n        Pad the exponent with zeros until it contains at least this\n        many digits. If omitted, the exponent will be at least 2 digits.\n    min_digits : non-negative integer or None, optional\n        Minimum number of digits to print. This only has an effect for\n        `unique=True`. In that case more digits than necessary to uniquely\n        identify the value may be printed and rounded unbiased.\n\n        .. versionadded:: 1.21.0\n\n    Returns\n    -------\n    rep : string\n        The string representation of the floating point value\n\n    See Also\n    --------\n    format_float_positional\n\n    Examples\n    --------\n    >>> np.format_float_scientific(np.float32(np.pi))\n    \'3.1415927e+00\'\n    >>> s = np.float32(1.23e24)\n    >>> np.format_float_scientific(s, unique=False, precision=15)\n    \'1.230000071797338e+24\'\n    >>> np.format_float_scientific(s, exp_digits=4)\n    \'1.23e+0024\'\n    '
    precision = _none_or_positive_arg(precision, 'precision')
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    exp_digits = _none_or_positive_arg(exp_digits, 'exp_digits')
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    if min_digits > 0 and precision > 0 and (min_digits > precision):
        raise ValueError('min_digits must be less than or equal to precision')
    return dragon4_scientific(x, precision=precision, unique=unique, trim=trim, sign=sign, pad_left=pad_left, exp_digits=exp_digits, min_digits=min_digits)

@set_module('numpy')
def format_float_positional(x, precision=None, unique=True, fractional=True, trim='k', sign=False, pad_left=None, pad_right=None, min_digits=None):
    if False:
        print('Hello World!')
    '\n    Format a floating-point scalar as a decimal string in positional notation.\n\n    Provides control over rounding, trimming and padding. Uses and assumes\n    IEEE unbiased rounding. Uses the "Dragon4" algorithm.\n\n    Parameters\n    ----------\n    x : python float or numpy floating scalar\n        Value to format.\n    precision : non-negative integer or None, optional\n        Maximum number of digits to print. May be None if `unique` is\n        `True`, but must be an integer if unique is `False`.\n    unique : boolean, optional\n        If `True`, use a digit-generation strategy which gives the shortest\n        representation which uniquely identifies the floating-point number from\n        other values of the same type, by judicious rounding. If `precision`\n        is given fewer digits than necessary can be printed, or if `min_digits`\n        is given more can be printed, in which cases the last digit is rounded\n        with unbiased rounding.\n        If `False`, digits are generated as if printing an infinite-precision\n        value and stopping after `precision` digits, rounding the remaining\n        value with unbiased rounding\n    fractional : boolean, optional\n        If `True`, the cutoffs of `precision` and `min_digits` refer to the\n        total number of digits after the decimal point, including leading\n        zeros.\n        If `False`, `precision` and `min_digits` refer to the total number of\n        significant digits, before or after the decimal point, ignoring leading\n        zeros.\n    trim : one of \'k\', \'.\', \'0\', \'-\', optional\n        Controls post-processing trimming of trailing digits, as follows:\n\n        * \'k\' : keep trailing zeros, keep decimal point (no trimming)\n        * \'.\' : trim all trailing zeros, leave decimal point\n        * \'0\' : trim all but the zero before the decimal point. Insert the\n          zero if it is missing.\n        * \'-\' : trim trailing zeros and any trailing decimal point\n    sign : boolean, optional\n        Whether to show the sign for positive values.\n    pad_left : non-negative integer, optional\n        Pad the left side of the string with whitespace until at least that\n        many characters are to the left of the decimal point.\n    pad_right : non-negative integer, optional\n        Pad the right side of the string with whitespace until at least that\n        many characters are to the right of the decimal point.\n    min_digits : non-negative integer or None, optional\n        Minimum number of digits to print. Only has an effect if `unique=True`\n        in which case additional digits past those necessary to uniquely\n        identify the value may be printed, rounding the last additional digit.\n\n        .. versionadded:: 1.21.0\n\n    Returns\n    -------\n    rep : string\n        The string representation of the floating point value\n\n    See Also\n    --------\n    format_float_scientific\n\n    Examples\n    --------\n    >>> np.format_float_positional(np.float32(np.pi))\n    \'3.1415927\'\n    >>> np.format_float_positional(np.float16(np.pi))\n    \'3.14\'\n    >>> np.format_float_positional(np.float16(0.3))\n    \'0.3\'\n    >>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)\n    \'0.3000488281\'\n    '
    precision = _none_or_positive_arg(precision, 'precision')
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    pad_right = _none_or_positive_arg(pad_right, 'pad_right')
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    if not fractional and precision == 0:
        raise ValueError('precision must be greater than 0 if fractional=False')
    if min_digits > 0 and precision > 0 and (min_digits > precision):
        raise ValueError('min_digits must be less than or equal to precision')
    return dragon4_positional(x, precision=precision, unique=unique, fractional=fractional, trim=trim, sign=sign, pad_left=pad_left, pad_right=pad_right, min_digits=min_digits)

class IntegerFormat:

    def __init__(self, data, sign='-'):
        if False:
            for i in range(10):
                print('nop')
        if data.size > 0:
            data_max = np.max(data)
            data_min = np.min(data)
            data_max_str_len = len(str(data_max))
            if sign == ' ' and data_min < 0:
                sign = '-'
            if data_max >= 0 and sign in '+ ':
                data_max_str_len += 1
            max_str_len = max(data_max_str_len, len(str(data_min)))
        else:
            max_str_len = 0
        self.format = f'{{:{sign}{max_str_len}d}}'

    def __call__(self, x):
        if False:
            while True:
                i = 10
        return self.format.format(x)

class BoolFormat:

    def __init__(self, data, **kwargs):
        if False:
            return 10
        self.truestr = ' True' if data.shape != () else 'True'

    def __call__(self, x):
        if False:
            return 10
        return self.truestr if x else 'False'

class ComplexFloatingFormat:
    """ Formatter for subtypes of np.complexfloating """

    def __init__(self, x, precision, floatmode, suppress_small, sign=False, *, legacy=None):
        if False:
            print('Hello World!')
        if isinstance(sign, bool):
            sign = '+' if sign else '-'
        floatmode_real = floatmode_imag = floatmode
        if legacy <= 113:
            floatmode_real = 'maxprec_equal'
            floatmode_imag = 'maxprec'
        self.real_format = FloatingFormat(x.real, precision, floatmode_real, suppress_small, sign=sign, legacy=legacy)
        self.imag_format = FloatingFormat(x.imag, precision, floatmode_imag, suppress_small, sign='+', legacy=legacy)

    def __call__(self, x):
        if False:
            while True:
                i = 10
        r = self.real_format(x.real)
        i = self.imag_format(x.imag)
        sp = len(i.rstrip())
        i = i[:sp] + 'j' + i[sp:]
        return r + i

class _TimelikeFormat:

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        non_nat = data[~isnat(data)]
        if len(non_nat) > 0:
            max_str_len = max(len(self._format_non_nat(np.max(non_nat))), len(self._format_non_nat(np.min(non_nat))))
        else:
            max_str_len = 0
        if len(non_nat) < data.size:
            max_str_len = max(max_str_len, 5)
        self._format = '%{}s'.format(max_str_len)
        self._nat = "'NaT'".rjust(max_str_len)

    def _format_non_nat(self, x):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        if isnat(x):
            return self._nat
        else:
            return self._format % self._format_non_nat(x)

class DatetimeFormat(_TimelikeFormat):

    def __init__(self, x, unit=None, timezone=None, casting='same_kind', legacy=False):
        if False:
            for i in range(10):
                print('nop')
        if unit is None:
            if x.dtype.kind == 'M':
                unit = datetime_data(x.dtype)[0]
            else:
                unit = 's'
        if timezone is None:
            timezone = 'naive'
        self.timezone = timezone
        self.unit = unit
        self.casting = casting
        self.legacy = legacy
        super().__init__(x)

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        if self.legacy <= 113:
            return self._format_non_nat(x)
        return super().__call__(x)

    def _format_non_nat(self, x):
        if False:
            for i in range(10):
                print('nop')
        return "'%s'" % datetime_as_string(x, unit=self.unit, timezone=self.timezone, casting=self.casting)

class TimedeltaFormat(_TimelikeFormat):

    def _format_non_nat(self, x):
        if False:
            return 10
        return str(x.astype('i8'))

class SubArrayFormat:

    def __init__(self, format_function, **options):
        if False:
            return 10
        self.format_function = format_function
        self.threshold = options['threshold']
        self.edge_items = options['edgeitems']

    def __call__(self, a):
        if False:
            return 10
        self.summary_insert = '...' if a.size > self.threshold else ''
        return self.format_array(a)

    def format_array(self, a):
        if False:
            while True:
                i = 10
        if np.ndim(a) == 0:
            return self.format_function(a)
        if self.summary_insert and a.shape[0] > 2 * self.edge_items:
            formatted = [self.format_array(a_) for a_ in a[:self.edge_items]] + [self.summary_insert] + [self.format_array(a_) for a_ in a[-self.edge_items:]]
        else:
            formatted = [self.format_array(a_) for a_ in a]
        return '[' + ', '.join(formatted) + ']'

class StructuredVoidFormat:
    """
    Formatter for structured np.void objects.

    This does not work on structured alias types like
    np.dtype(('i4', 'i2,i2')), as alias scalars lose their field information,
    and the implementation relies upon np.void.__getitem__.
    """

    def __init__(self, format_functions):
        if False:
            print('Hello World!')
        self.format_functions = format_functions

    @classmethod
    def from_data(cls, data, **options):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is a second way to initialize StructuredVoidFormat,\n        using the raw data as input. Added to avoid changing\n        the signature of __init__.\n        '
        format_functions = []
        for field_name in data.dtype.names:
            format_function = _get_format_function(data[field_name], **options)
            if data.dtype[field_name].shape != ():
                format_function = SubArrayFormat(format_function, **options)
            format_functions.append(format_function)
        return cls(format_functions)

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        str_fields = [format_function(field) for (field, format_function) in zip(x, self.format_functions)]
        if len(str_fields) == 1:
            return '({},)'.format(str_fields[0])
        else:
            return '({})'.format(', '.join(str_fields))

def _void_scalar_to_string(x, is_repr=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Implements the repr for structured-void scalars. It is called from the\n    scalartypes.c.src code, and is placed here because it uses the elementwise\n    formatters defined above.\n    '
    options = _format_options.copy()
    if options['legacy'] <= 125:
        return StructuredVoidFormat.from_data(array(x), **_format_options)(x)
    if options.get('formatter') is None:
        options['formatter'] = {}
    options['formatter'].setdefault('float_kind', str)
    val_repr = StructuredVoidFormat.from_data(array(x), **options)(x)
    if not is_repr:
        return val_repr
    cls = type(x)
    cls_fqn = cls.__module__.replace('numpy', 'np') + '.' + cls.__name__
    void_dtype = np.dtype((np.void, x.dtype))
    return f'{cls_fqn}({val_repr}, dtype={void_dtype!s})'
_typelessdata = [int_, float64, complex128, bool_]

def dtype_is_implied(dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine if the given dtype is implied by the representation\n    of its values.\n\n    Parameters\n    ----------\n    dtype : dtype\n        Data type\n\n    Returns\n    -------\n    implied : bool\n        True if the dtype is implied by the representation of its values.\n\n    Examples\n    --------\n    >>> np._core.arrayprint.dtype_is_implied(int)\n    True\n    >>> np.array([1, 2, 3], int)\n    array([1, 2, 3])\n    >>> np._core.arrayprint.dtype_is_implied(np.int8)\n    False\n    >>> np.array([1, 2, 3], np.int8)\n    array([1, 2, 3], dtype=int8)\n    '
    dtype = np.dtype(dtype)
    if _format_options['legacy'] <= 113 and dtype.type == bool_:
        return False
    if dtype.names is not None:
        return False
    if not dtype.isnative:
        return False
    return dtype.type in _typelessdata

def dtype_short_repr(dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a dtype to a short form which evaluates to the same dtype.\n\n    The intent is roughly that the following holds\n\n    >>> from numpy import *\n    >>> dt = np.int64([1, 2]).dtype\n    >>> assert eval(dtype_short_repr(dt)) == dt\n    '
    if type(dtype).__repr__ != np.dtype.__repr__:
        return repr(dtype)
    if dtype.names is not None:
        return str(dtype)
    elif issubclass(dtype.type, flexible):
        return "'%s'" % str(dtype)
    typename = dtype.name
    if not dtype.isnative:
        return "'%s'" % str(dtype)
    if typename and (not (typename[0].isalpha() and typename.isalnum())):
        typename = repr(typename)
    return typename

def _array_repr_implementation(arr, max_line_width=None, precision=None, suppress_small=None, array2string=array2string):
    if False:
        return 10
    'Internal version of array_repr() that allows overriding array2string.'
    if max_line_width is None:
        max_line_width = _format_options['linewidth']
    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = 'array'
    skipdtype = dtype_is_implied(arr.dtype) and arr.size > 0
    prefix = class_name + '('
    suffix = ')' if skipdtype else ','
    if _format_options['legacy'] <= 113 and arr.shape == () and (not arr.dtype.names):
        lst = repr(arr.item())
    elif arr.size > 0 or arr.shape == (0,):
        lst = array2string(arr, max_line_width, precision, suppress_small, ', ', prefix, suffix=suffix)
    else:
        lst = '[], shape=%s' % (repr(arr.shape),)
    arr_str = prefix + lst + suffix
    if skipdtype:
        return arr_str
    dtype_str = 'dtype={})'.format(dtype_short_repr(arr.dtype))
    last_line_len = len(arr_str) - (arr_str.rfind('\n') + 1)
    spacer = ' '
    if _format_options['legacy'] <= 113:
        if issubclass(arr.dtype.type, flexible):
            spacer = '\n' + ' ' * len(class_name + '(')
    elif last_line_len + len(dtype_str) + 1 > max_line_width:
        spacer = '\n' + ' ' * len(class_name + '(')
    return arr_str + spacer + dtype_str

def _array_repr_dispatcher(arr, max_line_width=None, precision=None, suppress_small=None):
    if False:
        return 10
    return (arr,)

@array_function_dispatch(_array_repr_dispatcher, module='numpy')
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    if False:
        i = 10
        return i + 15
    '\n    Return the string representation of an array.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array.\n    max_line_width : int, optional\n        Inserts newlines if text is longer than `max_line_width`.\n        Defaults to ``numpy.get_printoptions()[\'linewidth\']``.\n    precision : int, optional\n        Floating point precision.\n        Defaults to ``numpy.get_printoptions()[\'precision\']``.\n    suppress_small : bool, optional\n        Represent numbers "very close" to zero as zero; default is False.\n        Very close is defined by precision: if the precision is 8, e.g.,\n        numbers smaller (in absolute value) than 5e-9 are represented as\n        zero.\n        Defaults to ``numpy.get_printoptions()[\'suppress\']``.\n\n    Returns\n    -------\n    string : str\n      The string representation of an array.\n\n    See Also\n    --------\n    array_str, array2string, set_printoptions\n\n    Examples\n    --------\n    >>> np.array_repr(np.array([1,2]))\n    \'array([1, 2])\'\n    >>> np.array_repr(np.ma.array([0.]))\n    \'MaskedArray([0.])\'\n    >>> np.array_repr(np.array([], np.int32))\n    \'array([], dtype=int32)\'\n\n    >>> x = np.array([1e-6, 4e-7, 2, 3])\n    >>> np.array_repr(x, precision=6, suppress_small=True)\n    \'array([0.000001,  0.      ,  2.      ,  3.      ])\'\n\n    '
    return _array_repr_implementation(arr, max_line_width, precision, suppress_small)

@_recursive_guard()
def _guarded_repr_or_str(v):
    if False:
        while True:
            i = 10
    if isinstance(v, bytes):
        return repr(v)
    return str(v)

def _array_str_implementation(a, max_line_width=None, precision=None, suppress_small=None, array2string=array2string):
    if False:
        i = 10
        return i + 15
    'Internal version of array_str() that allows overriding array2string.'
    if _format_options['legacy'] <= 113 and a.shape == () and (not a.dtype.names):
        return str(a.item())
    if a.shape == ():
        return _guarded_repr_or_str(np.ndarray.__getitem__(a, ()))
    return array2string(a, max_line_width, precision, suppress_small, ' ', '')

def _array_str_dispatcher(a, max_line_width=None, precision=None, suppress_small=None):
    if False:
        while True:
            i = 10
    return (a,)

@array_function_dispatch(_array_str_dispatcher, module='numpy')
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a string representation of the data in an array.\n\n    The data in the array is returned as a single string.  This function is\n    similar to `array_repr`, the difference being that `array_repr` also\n    returns information on the kind of array and its data type.\n\n    Parameters\n    ----------\n    a : ndarray\n        Input array.\n    max_line_width : int, optional\n        Inserts newlines if text is longer than `max_line_width`.\n        Defaults to ``numpy.get_printoptions()[\'linewidth\']``.\n    precision : int, optional\n        Floating point precision.\n        Defaults to ``numpy.get_printoptions()[\'precision\']``.\n    suppress_small : bool, optional\n        Represent numbers "very close" to zero as zero; default is False.\n        Very close is defined by precision: if the precision is 8, e.g.,\n        numbers smaller (in absolute value) than 5e-9 are represented as\n        zero.\n        Defaults to ``numpy.get_printoptions()[\'suppress\']``.\n\n    See Also\n    --------\n    array2string, array_repr, set_printoptions\n\n    Examples\n    --------\n    >>> np.array_str(np.arange(3))\n    \'[0 1 2]\'\n\n    '
    return _array_str_implementation(a, max_line_width, precision, suppress_small)
_array2string_impl = getattr(array2string, '__wrapped__', array2string)
_default_array_str = functools.partial(_array_str_implementation, array2string=_array2string_impl)
_default_array_repr = functools.partial(_array_repr_implementation, array2string=_array2string_impl)

def set_string_function(f, repr=True):
    if False:
        i = 10
        return i + 15
    "\n    Set a Python function to be used when pretty printing arrays.\n\n    .. deprecated:: 2.0\n        Use `np.set_printoptions` instead with a formatter for custom\n        printing of NumPy objects.\n\n    Parameters\n    ----------\n    f : function or None\n        Function to be used to pretty print arrays. The function should expect\n        a single array argument and return a string of the representation of\n        the array. If None, the function is reset to the default NumPy function\n        to print arrays.\n    repr : bool, optional\n        If True (default), the function for pretty printing (``__repr__``)\n        is set, if False the function that returns the default string\n        representation (``__str__``) is set.\n\n    See Also\n    --------\n    set_printoptions, get_printoptions\n\n    Examples\n    --------\n    >>> from numpy._core.arrayprint import set_string_function\n    >>> def pprint(arr):\n    ...     return 'HA! - What are you going to do now?'\n    ...\n    >>> set_string_function(pprint)\n    >>> a = np.arange(10)\n    >>> a\n    HA! - What are you going to do now?\n    >>> _ = a\n    >>> # [0 1 2 3 4 5 6 7 8 9]\n\n    We can reset the function to the default:\n\n    >>> set_string_function(None)\n    >>> a\n    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n\n    `repr` affects either pretty printing or normal string representation.\n    Note that ``__repr__`` is still affected by setting ``__str__``\n    because the width of each array element in the returned string becomes\n    equal to the length of the result of ``__str__()``.\n\n    >>> x = np.arange(4)\n    >>> set_string_function(lambda x:'random', repr=False)\n    >>> x.__str__()\n    'random'\n    >>> x.__repr__()\n    'array([0, 1, 2, 3])'\n\n    "
    warnings.warn('`set_string_function` is deprecated. Use `np.set_printoptions` with a formatter for custom printing NumPy objects. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    if f is None:
        if repr:
            return multiarray.set_string_function(_default_array_repr, 1)
        else:
            return multiarray.set_string_function(_default_array_str, 0)
    else:
        return multiarray.set_string_function(f, repr)