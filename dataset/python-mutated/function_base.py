import functools
import warnings
import operator
import types
import numpy as np
from . import numeric as _nx
from .numeric import result_type, nan, asanyarray, ndim
from numpy._core.multiarray import add_docstring
from numpy._core import overrides
__all__ = ['logspace', 'linspace', 'geomspace']
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')

def _linspace_dispatcher(start, stop, num=None, endpoint=None, retstep=None, dtype=None, axis=None):
    if False:
        print('Hello World!')
    return (start, stop)

@array_function_dispatch(_linspace_dispatcher)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    if False:
        while True:
            i = 10
    "\n    Return evenly spaced numbers over a specified interval.\n\n    Returns `num` evenly spaced samples, calculated over the\n    interval [`start`, `stop`].\n\n    The endpoint of the interval can optionally be excluded.\n\n    .. versionchanged:: 1.16.0\n        Non-scalar `start` and `stop` are now supported.\n\n    .. versionchanged:: 1.20.0\n        Values are rounded towards ``-inf`` instead of ``0`` when an\n        integer ``dtype`` is specified. The old behavior can\n        still be obtained with ``np.linspace(start, stop, num).astype(int)``\n\n    Parameters\n    ----------\n    start : array_like\n        The starting value of the sequence.\n    stop : array_like\n        The end value of the sequence, unless `endpoint` is set to False.\n        In that case, the sequence consists of all but the last of ``num + 1``\n        evenly spaced samples, so that `stop` is excluded.  Note that the step\n        size changes when `endpoint` is False.\n    num : int, optional\n        Number of samples to generate. Default is 50. Must be non-negative.\n    endpoint : bool, optional\n        If True, `stop` is the last sample. Otherwise, it is not included.\n        Default is True.\n    retstep : bool, optional\n        If True, return (`samples`, `step`), where `step` is the spacing\n        between samples.\n    dtype : dtype, optional\n        The type of the output array.  If `dtype` is not given, the data type\n        is inferred from `start` and `stop`. The inferred dtype will never be\n        an integer; `float` is chosen even if the arguments would produce an\n        array of integers.\n\n        .. versionadded:: 1.9.0\n\n    axis : int, optional\n        The axis in the result to store the samples.  Relevant only if start\n        or stop are array-like.  By default (0), the samples will be along a\n        new axis inserted at the beginning. Use -1 to get an axis at the end.\n\n        .. versionadded:: 1.16.0\n\n    Returns\n    -------\n    samples : ndarray\n        There are `num` equally spaced samples in the closed interval\n        ``[start, stop]`` or the half-open interval ``[start, stop)``\n        (depending on whether `endpoint` is True or False).\n    step : float, optional\n        Only returned if `retstep` is True\n\n        Size of spacing between samples.\n\n\n    See Also\n    --------\n    arange : Similar to `linspace`, but uses a step size (instead of the\n             number of samples).\n    geomspace : Similar to `linspace`, but with numbers spaced evenly on a log\n                scale (a geometric progression).\n    logspace : Similar to `geomspace`, but with the end points specified as\n               logarithms.\n    :ref:`how-to-partition`\n\n    Examples\n    --------\n    >>> np.linspace(2.0, 3.0, num=5)\n    array([2.  , 2.25, 2.5 , 2.75, 3.  ])\n    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)\n    array([2. ,  2.2,  2.4,  2.6,  2.8])\n    >>> np.linspace(2.0, 3.0, num=5, retstep=True)\n    (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)\n\n    Graphical illustration:\n\n    >>> import matplotlib.pyplot as plt\n    >>> N = 8\n    >>> y = np.zeros(N)\n    >>> x1 = np.linspace(0, 10, N, endpoint=True)\n    >>> x2 = np.linspace(0, 10, N, endpoint=False)\n    >>> plt.plot(x1, y, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.plot(x2, y + 0.5, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.ylim([-0.5, 1])\n    (-0.5, 1)\n    >>> plt.show()\n\n    "
    num = operator.index(num)
    if num < 0:
        raise ValueError('Number of samples, %s, must be non-negative.' % num)
    div = num - 1 if endpoint else num
    start = asanyarray(start) * 1.0
    stop = asanyarray(stop) * 1.0
    dt = result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt
        integer_dtype = False
    else:
        integer_dtype = _nx.issubdtype(dtype, _nx.integer)
    delta = stop - start
    y = _nx.arange(0, num, dtype=dt).reshape((-1,) + (1,) * ndim(delta))
    if div > 0:
        _mult_inplace = _nx.isscalar(delta)
        step = delta / div
        any_step_zero = step == 0 if _mult_inplace else _nx.asanyarray(step == 0).any()
        if any_step_zero:
            y /= div
            if _mult_inplace:
                y *= delta
            else:
                y = y * delta
        elif _mult_inplace:
            y *= step
        else:
            y = y * step
    else:
        step = nan
        y = y * delta
    y += start
    if endpoint and num > 1:
        y[-1, ...] = stop
    if axis != 0:
        y = _nx.moveaxis(y, 0, axis)
    if integer_dtype:
        _nx.floor(y, out=y)
    if retstep:
        return (y.astype(dtype, copy=False), step)
    else:
        return y.astype(dtype, copy=False)

def _logspace_dispatcher(start, stop, num=None, endpoint=None, base=None, dtype=None, axis=None):
    if False:
        for i in range(10):
            print('nop')
    return (start, stop, base)

@array_function_dispatch(_logspace_dispatcher)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if False:
        i = 10
        return i + 15
    "\n    Return numbers spaced evenly on a log scale.\n\n    In linear space, the sequence starts at ``base ** start``\n    (`base` to the power of `start`) and ends with ``base ** stop``\n    (see `endpoint` below).\n\n    .. versionchanged:: 1.16.0\n        Non-scalar `start` and `stop` are now supported.\n\n    .. versionchanged:: 1.25.0\n        Non-scalar 'base` is now supported\n\n    Parameters\n    ----------\n    start : array_like\n        ``base ** start`` is the starting value of the sequence.\n    stop : array_like\n        ``base ** stop`` is the final value of the sequence, unless `endpoint`\n        is False.  In that case, ``num + 1`` values are spaced over the\n        interval in log-space, of which all but the last (a sequence of\n        length `num`) are returned.\n    num : integer, optional\n        Number of samples to generate.  Default is 50.\n    endpoint : boolean, optional\n        If true, `stop` is the last sample. Otherwise, it is not included.\n        Default is True.\n    base : array_like, optional\n        The base of the log space. The step size between the elements in\n        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.\n        Default is 10.0.\n    dtype : dtype\n        The type of the output array.  If `dtype` is not given, the data type\n        is inferred from `start` and `stop`. The inferred type will never be\n        an integer; `float` is chosen even if the arguments would produce an\n        array of integers.\n    axis : int, optional\n        The axis in the result to store the samples.  Relevant only if start,\n        stop, or base are array-like.  By default (0), the samples will be\n        along a new axis inserted at the beginning. Use -1 to get an axis at\n        the end.\n\n        .. versionadded:: 1.16.0\n\n\n    Returns\n    -------\n    samples : ndarray\n        `num` samples, equally spaced on a log scale.\n\n    See Also\n    --------\n    arange : Similar to linspace, with the step size specified instead of the\n             number of samples. Note that, when used with a float endpoint, the\n             endpoint may or may not be included.\n    linspace : Similar to logspace, but with the samples uniformly distributed\n               in linear space, instead of log space.\n    geomspace : Similar to logspace, but with endpoints specified directly.\n    :ref:`how-to-partition`\n\n    Notes\n    -----\n    If base is a scalar, logspace is equivalent to the code\n\n    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)\n    ... # doctest: +SKIP\n    >>> power(base, y).astype(dtype)\n    ... # doctest: +SKIP\n\n    Examples\n    --------\n    >>> np.logspace(2.0, 3.0, num=4)\n    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])\n    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)\n    array([100.        ,  177.827941  ,  316.22776602,  562.34132519])\n    >>> np.logspace(2.0, 3.0, num=4, base=2.0)\n    array([4.        ,  5.0396842 ,  6.34960421,  8.        ])\n    >>> np.logspace(2.0, 3.0, num=4, base=[2.0, 3.0], axis=-1)\n    array([[ 4.        ,  5.0396842 ,  6.34960421,  8.        ],\n           [ 9.        , 12.98024613, 18.72075441, 27.        ]])\n\n    Graphical illustration:\n\n    >>> import matplotlib.pyplot as plt\n    >>> N = 10\n    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)\n    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)\n    >>> y = np.zeros(N)\n    >>> plt.plot(x1, y, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.plot(x2, y + 0.5, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.ylim([-0.5, 1])\n    (-0.5, 1)\n    >>> plt.show()\n\n    "
    ndmax = np.broadcast(start, stop, base).ndim
    (start, stop, base) = (np.array(a, copy=False, subok=True, ndmin=ndmax) for a in (start, stop, base))
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    base = np.expand_dims(base, axis=axis)
    if dtype is None:
        return _nx.power(base, y)
    return _nx.power(base, y).astype(dtype, copy=False)

def _geomspace_dispatcher(start, stop, num=None, endpoint=None, dtype=None, axis=None):
    if False:
        i = 10
        return i + 15
    return (start, stop)

@array_function_dispatch(_geomspace_dispatcher)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if False:
        return 10
    "\n    Return numbers spaced evenly on a log scale (a geometric progression).\n\n    This is similar to `logspace`, but with endpoints specified directly.\n    Each output sample is a constant multiple of the previous.\n\n    .. versionchanged:: 1.16.0\n        Non-scalar `start` and `stop` are now supported.\n\n    Parameters\n    ----------\n    start : array_like\n        The starting value of the sequence.\n    stop : array_like\n        The final value of the sequence, unless `endpoint` is False.\n        In that case, ``num + 1`` values are spaced over the\n        interval in log-space, of which all but the last (a sequence of\n        length `num`) are returned.\n    num : integer, optional\n        Number of samples to generate.  Default is 50.\n    endpoint : boolean, optional\n        If true, `stop` is the last sample. Otherwise, it is not included.\n        Default is True.\n    dtype : dtype\n        The type of the output array.  If `dtype` is not given, the data type\n        is inferred from `start` and `stop`. The inferred dtype will never be\n        an integer; `float` is chosen even if the arguments would produce an\n        array of integers.\n    axis : int, optional\n        The axis in the result to store the samples.  Relevant only if start\n        or stop are array-like.  By default (0), the samples will be along a\n        new axis inserted at the beginning. Use -1 to get an axis at the end.\n\n        .. versionadded:: 1.16.0\n\n    Returns\n    -------\n    samples : ndarray\n        `num` samples, equally spaced on a log scale.\n\n    See Also\n    --------\n    logspace : Similar to geomspace, but with endpoints specified using log\n               and base.\n    linspace : Similar to geomspace, but with arithmetic instead of geometric\n               progression.\n    arange : Similar to linspace, with the step size specified instead of the\n             number of samples.\n    :ref:`how-to-partition`\n\n    Notes\n    -----\n    If the inputs or dtype are complex, the output will follow a logarithmic\n    spiral in the complex plane.  (There are an infinite number of spirals\n    passing through two points; the output will follow the shortest such path.)\n\n    Examples\n    --------\n    >>> np.geomspace(1, 1000, num=4)\n    array([    1.,    10.,   100.,  1000.])\n    >>> np.geomspace(1, 1000, num=3, endpoint=False)\n    array([   1.,   10.,  100.])\n    >>> np.geomspace(1, 1000, num=4, endpoint=False)\n    array([   1.        ,    5.62341325,   31.6227766 ,  177.827941  ])\n    >>> np.geomspace(1, 256, num=9)\n    array([   1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.])\n\n    Note that the above may not produce exact integers:\n\n    >>> np.geomspace(1, 256, num=9, dtype=int)\n    array([  1,   2,   4,   7,  16,  32,  63, 127, 256])\n    >>> np.around(np.geomspace(1, 256, num=9)).astype(int)\n    array([  1,   2,   4,   8,  16,  32,  64, 128, 256])\n\n    Negative, decreasing, and complex inputs are allowed:\n\n    >>> np.geomspace(1000, 1, num=4)\n    array([1000.,  100.,   10.,    1.])\n    >>> np.geomspace(-1000, -1, num=4)\n    array([-1000.,  -100.,   -10.,    -1.])\n    >>> np.geomspace(1j, 1000j, num=4)  # Straight line\n    array([0.   +1.j, 0.  +10.j, 0. +100.j, 0.+1000.j])\n    >>> np.geomspace(-1+0j, 1+0j, num=5)  # Circle\n    array([-1.00000000e+00+1.22464680e-16j, -7.07106781e-01+7.07106781e-01j,\n            6.12323400e-17+1.00000000e+00j,  7.07106781e-01+7.07106781e-01j,\n            1.00000000e+00+0.00000000e+00j])\n\n    Graphical illustration of `endpoint` parameter:\n\n    >>> import matplotlib.pyplot as plt\n    >>> N = 10\n    >>> y = np.zeros(N)\n    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=True), y + 1, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=False), y + 2, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.axis([0.5, 2000, 0, 3])\n    [0.5, 2000, 0, 3]\n    >>> plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')\n    >>> plt.show()\n\n    "
    start = asanyarray(start)
    stop = asanyarray(stop)
    if _nx.any(start == 0) or _nx.any(stop == 0):
        raise ValueError('Geometric sequence cannot include zero')
    dt = result_type(start, stop, float(num), _nx.zeros((), dtype))
    if dtype is None:
        dtype = dt
    else:
        dtype = _nx.dtype(dtype)
    start = start.astype(dt, copy=True)
    stop = stop.astype(dt, copy=True)
    out_sign = _nx.ones(_nx.broadcast(start, stop).shape, dt)
    if _nx.issubdtype(dt, _nx.complexfloating):
        all_imag = (start.real == 0.0) & (stop.real == 0.0)
        if _nx.any(all_imag):
            start[all_imag] = start[all_imag].imag
            stop[all_imag] = stop[all_imag].imag
            out_sign[all_imag] = 1j
    both_negative = (_nx.sign(start) == -1) & (_nx.sign(stop) == -1)
    if _nx.any(both_negative):
        _nx.negative(start, out=start, where=both_negative)
        _nx.negative(stop, out=stop, where=both_negative)
        _nx.negative(out_sign, out=out_sign, where=both_negative)
    log_start = _nx.log10(start)
    log_stop = _nx.log10(stop)
    result = logspace(log_start, log_stop, num=num, endpoint=endpoint, base=10.0, dtype=dtype)
    if num > 0:
        result[0] = start
        if num > 1 and endpoint:
            result[-1] = stop
    result = out_sign * result
    if axis != 0:
        result = _nx.moveaxis(result, 0, axis)
    return result.astype(dtype, copy=False)

def _needs_add_docstring(obj):
    if False:
        return 10
    '\n    Returns true if the only way to set the docstring of `obj` from python is\n    via add_docstring.\n\n    This function errs on the side of being overly conservative.\n    '
    Py_TPFLAGS_HEAPTYPE = 1 << 9
    if isinstance(obj, (types.FunctionType, types.MethodType, property)):
        return False
    if isinstance(obj, type) and obj.__flags__ & Py_TPFLAGS_HEAPTYPE:
        return False
    return True

def _add_docstring(obj, doc, warn_on_python):
    if False:
        for i in range(10):
            print('nop')
    if warn_on_python and (not _needs_add_docstring(obj)):
        warnings.warn('add_newdoc was used on a pure-python object {}. Prefer to attach it directly to the source.'.format(obj), UserWarning, stacklevel=3)
    try:
        add_docstring(obj, doc)
    except Exception:
        pass

def add_newdoc(place, obj, doc, warn_on_python=True):
    if False:
        return 10
    "\n    Add documentation to an existing object, typically one defined in C\n\n    The purpose is to allow easier editing of the docstrings without requiring\n    a re-compile. This exists primarily for internal use within numpy itself.\n\n    Parameters\n    ----------\n    place : str\n        The absolute name of the module to import from\n    obj : str or None\n        The name of the object to add documentation to, typically a class or\n        function name.\n    doc : {str, Tuple[str, str], List[Tuple[str, str]]}\n        If a string, the documentation to apply to `obj`\n\n        If a tuple, then the first element is interpreted as an attribute\n        of `obj` and the second as the docstring to apply -\n        ``(method, docstring)``\n\n        If a list, then each element of the list should be a tuple of length\n        two - ``[(method1, docstring1), (method2, docstring2), ...]``\n    warn_on_python : bool\n        If True, the default, emit `UserWarning` if this is used to attach\n        documentation to a pure-python object.\n\n    Notes\n    -----\n    This routine never raises an error if the docstring can't be written, but\n    will raise an error if the object being documented does not exist.\n\n    This routine cannot modify read-only docstrings, as appear\n    in new-style classes or built-in functions. Because this\n    routine never raises an error the caller must check manually\n    that the docstrings were changed.\n\n    Since this function grabs the ``char *`` from a c-level str object and puts\n    it into the ``tp_doc`` slot of the type of `obj`, it violates a number of\n    C-API best-practices, by:\n\n    - modifying a `PyTypeObject` after calling `PyType_Ready`\n    - calling `Py_INCREF` on the str and losing the reference, so the str\n      will never be released\n\n    If possible it should be avoided.\n    "
    new = getattr(__import__(place, globals(), {}, [obj]), obj)
    if isinstance(doc, str):
        _add_docstring(new, doc.strip(), warn_on_python)
    elif isinstance(doc, tuple):
        (attr, docstring) = doc
        _add_docstring(getattr(new, attr), docstring.strip(), warn_on_python)
    elif isinstance(doc, list):
        for (attr, docstring) in doc:
            _add_docstring(getattr(new, attr), docstring.strip(), warn_on_python)