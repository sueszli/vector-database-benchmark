import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
if is_available():
    import _pytest.outcomes
    _is_pytest_available = True
    _skip_classes: Tuple[Type, ...] = (unittest.SkipTest, _pytest.outcomes.Skipped)
else:
    _is_pytest_available = False
    _skip_classes = (unittest.SkipTest,)

def _format_exception(exc):
    if False:
        print('Hello World!')
    if exc is None:
        return None
    return ''.join(traceback.TracebackException.from_exception(exc).format())

def _call_func(impl, args, kw):
    if False:
        return 10
    exceptions = (Exception,)
    if _is_pytest_available:
        exceptions += (_pytest.outcomes.Skipped,)
    try:
        result = impl(*args, **kw)
        error = None
    except exceptions as e:
        tb = e.__traceback__
        if tb.tb_next is None:
            raise e
        result = None
        error = e
    return (result, error)

def _call_func_cupy(impl, args, kw, name, sp_name, scipy_name):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)
    kw = kw.copy()
    if sp_name:
        kw[sp_name] = cupyx.scipy.sparse
    if scipy_name:
        kw[scipy_name] = cupyx.scipy
    kw[name] = cupy
    (result, error) = _call_func(impl, args, kw)
    return (result, error)

def _call_func_numpy(impl, args, kw, name, sp_name, scipy_name):
    if False:
        print('Hello World!')
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)
    kw = kw.copy()
    kw[name] = numpy
    if sp_name:
        import scipy.sparse
        kw[sp_name] = scipy.sparse
    if scipy_name:
        import scipy
        kw[scipy_name] = scipy
    (result, error) = _call_func(impl, args, kw)
    return (result, error)

def _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name):
    if False:
        print('Hello World!')
    (cupy_result, cupy_error) = _call_func_cupy(impl, args, kw, name, sp_name, scipy_name)
    (numpy_result, numpy_error) = _call_func_numpy(impl, args, kw, name, sp_name, scipy_name)
    return (cupy_result, cupy_error, numpy_result, numpy_error)
_numpy_errors = [AttributeError, Exception, IndexError, TypeError, ValueError, NotImplementedError, DeprecationWarning, numpy.AxisError, numpy.linalg.LinAlgError]

def _check_numpy_cupy_error_compatible(cupy_error, numpy_error):
    if False:
        for i in range(10):
            print('nop')
    'Checks if try/except blocks are equivalent up to public error classes\n    '
    return all((isinstance(cupy_error, err) == isinstance(numpy_error, err) for err in _numpy_errors))

def _fail_test_with_unexpected_errors(tb, msg_format, cupy_error, numpy_error):
    if False:
        return 10
    msg = msg_format.format(cupy_error=_format_exception(cupy_error), numpy_error=_format_exception(numpy_error))
    raise AssertionError(msg).with_traceback(tb)

def _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=False):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(cupy_error, _skip_classes) and isinstance(numpy_error, _skip_classes):
        if cupy_error.__class__ is not numpy_error.__class__:
            raise AssertionError('Both numpy and cupy were skipped but with different exceptions.')
        if cupy_error.args != numpy_error.args:
            raise AssertionError('Both numpy and cupy were skipped but with different causes.')
        raise numpy_error
    if os.environ.get('CUPY_CI', '') != '' and cupy_error is not None:
        frame = traceback.extract_tb(cupy_error.__traceback__)[-1]
        filename = os.path.basename(frame.filename)
        if filename == 'test_helper.py':
            pass
        elif filename.startswith('test_'):
            _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Error was raised from test code.\n\n{cupy_error}', cupy_error, None)
    if accept_error is True:
        accept_error = Exception
    elif not accept_error:
        accept_error = ()
    if cupy_error is None and numpy_error is None:
        raise AssertionError('Both cupy and numpy are expected to raise errors, but not')
    elif cupy_error is None:
        _fail_test_with_unexpected_errors(numpy_error.__traceback__, 'Only numpy raises error\n\n{numpy_error}', None, numpy_error)
    elif numpy_error is None:
        _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Only cupy raises error\n\n{cupy_error}', cupy_error, None)
    elif not _check_numpy_cupy_error_compatible(cupy_error, numpy_error):
        _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Different types of errors occurred\n\ncupy\n{cupy_error}\n\nnumpy\n{numpy_error}\n', cupy_error, numpy_error)
    elif not (isinstance(cupy_error, accept_error) and isinstance(numpy_error, accept_error)):
        _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Both cupy and numpy raise exceptions\n\ncupy\n{cupy_error}\n\nnumpy\n{numpy_error}\n', cupy_error, numpy_error)

def _signed_counterpart(dtype):
    if False:
        return 10
    return numpy.dtype(numpy.dtype(dtype).char.lower()).type

def _make_positive_masks(impl, args, kw, name, sp_name, scipy_name):
    if False:
        return 10
    ks = [k for (k, v) in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = _signed_counterpart(kw[k])
    (result, error) = _call_func_cupy(impl, args, kw, name, sp_name, scipy_name)
    assert error is None
    if not isinstance(result, (tuple, list)):
        result = (result,)
    return [cupy.asnumpy(r) >= 0 for r in result]

def _contains_signed_and_unsigned(kw):
    if False:
        print('Hello World!')

    def isdtype(v):
        if False:
            print('Hello World!')
        if isinstance(v, numpy.dtype):
            return True
        elif isinstance(v, str):
            return True
        elif isinstance(v, type) and issubclass(v, numpy.number):
            return True
        else:
            return False
    vs = set((v for v in kw.values() if isdtype(v)))
    return any((d in vs for d in _unsigned_dtypes)) and any((d in vs for d in _float_dtypes + _signed_dtypes))

def _wraps_partial(wrapped, *names):
    if False:
        while True:
            i = 10

    def decorator(impl):
        if False:
            print('Hello World!')
        impl = functools.wraps(wrapped)(impl)
        impl.__signature__ = inspect.signature(functools.partial(wrapped, **{name: None for name in names}))
        return impl
    return decorator

def _wraps_partial_xp(wrapped, name, sp_name, scipy_name):
    if False:
        i = 10
        return i + 15
    names = [name, sp_name, scipy_name]
    names = [n for n in names if n is not None]
    return _wraps_partial(wrapped, *names)

def _make_decorator(check_func, name, type_check, contiguous_check, accept_error, sp_name=None, scipy_name=None, check_sparse_format=True):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)

    def decorator(impl):
        if False:
            return 10

        @_wraps_partial_xp(impl, name, sp_name, scipy_name)
        def test_func(*args, **kw):
            if False:
                print('Hello World!')
            (cupy_result, cupy_error, numpy_result, numpy_error) = _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name)
            assert cupy_result is not None or cupy_error is not None
            assert numpy_result is not None or numpy_error is not None
            if cupy_error or numpy_error:
                _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=accept_error)
                return
            if not isinstance(cupy_result, (tuple, list)):
                cupy_result = (cupy_result,)
            if not isinstance(numpy_result, (tuple, list)):
                numpy_result = (numpy_result,)
            assert len(cupy_result) == len(numpy_result)
            cupy_numpy_result_ndarrays = [_convert_output_to_ndarray(cupy_r, numpy_r, sp_name, check_sparse_format) for (cupy_r, numpy_r) in zip(cupy_result, numpy_result)]
            if type_check:
                for (cupy_r, numpy_r) in cupy_numpy_result_ndarrays:
                    if cupy_r.dtype != numpy_r.dtype:
                        raise AssertionError('ndarrays of different dtypes are returned.\ncupy: {}\nnumpy: {}'.format(cupy_r.dtype, numpy_r.dtype))
            if contiguous_check:
                for (cupy_r, numpy_r) in zip(cupy_result, numpy_result):
                    if isinstance(numpy_r, numpy.ndarray):
                        if numpy_r.flags.c_contiguous and (not cupy_r.flags.c_contiguous):
                            raise AssertionError('The state of c_contiguous flag is false. (cupy_result:{} numpy_result:{})'.format(cupy_r.flags.c_contiguous, numpy_r.flags.c_contiguous))
                        if numpy_r.flags.f_contiguous and (not cupy_r.flags.f_contiguous):
                            raise AssertionError('The state of f_contiguous flag is false. (cupy_result:{} numpy_result:{})'.format(cupy_r.flags.f_contiguous, numpy_r.flags.f_contiguous))
            for (cupy_r, numpy_r) in cupy_numpy_result_ndarrays:
                assert cupy_r.shape == numpy_r.shape
            masks = [None] * len(cupy_result)
            if _contains_signed_and_unsigned(kw):
                needs_mask = [cupy_r.dtype in _unsigned_dtypes for cupy_r in cupy_result]
                if any(needs_mask):
                    masks = _make_positive_masks(impl, args, kw, name, sp_name, scipy_name)
                    for (i, flag) in enumerate(needs_mask):
                        if not flag:
                            masks[i] = None
            for ((cupy_r, numpy_r), mask) in zip(cupy_numpy_result_ndarrays, masks):
                skip = False
                if mask is not None:
                    if cupy_r.shape == ():
                        skip = (mask == 0).all()
                    else:
                        cupy_r = cupy_r[mask].get()
                        numpy_r = numpy_r[mask]
                if not skip:
                    check_func(cupy_r, numpy_r)
        return test_func
    return decorator

def _convert_output_to_ndarray(c_out, n_out, sp_name, check_sparse_format):
    if False:
        for i in range(10):
            print('nop')
    'Checks type of cupy/numpy results and returns cupy/numpy ndarrays.\n\n    Args:\n        c_out (cupy.ndarray, cupyx.scipy.sparse matrix, cupy.poly1d or scalar):\n            cupy result\n        n_out (numpy.ndarray, scipy.sparse matrix, numpy.poly1d or scalar):\n            numpy result\n        sp_name(str or None): Argument name whose value is either\n            ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n            argument is given for the modules.\n        check_sparse_format (bool): If ``True``, consistency of format of\n            sparse matrix is also checked. Default is ``True``.\n\n    Returns:\n        The tuple of cupy.ndarray and numpy.ndarray.\n    '
    if sp_name is not None and cupyx.scipy.sparse.issparse(c_out):
        import scipy.sparse
        assert scipy.sparse.issparse(n_out)
        if check_sparse_format:
            assert c_out.format == n_out.format
        return (c_out.A, n_out.A)
    if isinstance(c_out, cupy.ndarray) and isinstance(n_out, (numpy.ndarray, numpy.generic)):
        return (c_out, n_out)
    if isinstance(c_out, cupy.poly1d) and isinstance(n_out, numpy.poly1d):
        assert c_out.variable == n_out.variable
        return (c_out.coeffs, n_out.coeffs)
    if isinstance(c_out, numpy.generic) and isinstance(n_out, numpy.generic):
        return (c_out, n_out)
    if numpy.isscalar(c_out) and numpy.isscalar(n_out):
        return (cupy.array(c_out), numpy.array(n_out))
    raise AssertionError('numpy and cupy returns different type of return value:\ncupy: {}\nnumpy: {}'.format(type(c_out), type(n_out)))

def _check_tolerance_keys(rtol, atol):
    if False:
        for i in range(10):
            print('nop')

    def _check(tol):
        if False:
            while True:
                i = 10
        if isinstance(tol, dict):
            for k in tol.keys():
                if type(k) is type:
                    continue
                if type(k) is str and k == 'default':
                    continue
                msg = "Keys of the tolerance dictionary need to be type objects as `numpy.float32` and `cupy.float32` or `'default'` string."
                raise TypeError(msg)
    _check(rtol)
    _check(atol)

def _resolve_tolerance(type_check, result, rtol, atol):
    if False:
        return 10

    def _resolve(dtype, tol):
        if False:
            while True:
                i = 10
        if isinstance(tol, dict):
            tol1 = tol.get(dtype.type)
            if tol1 is None:
                tol1 = tol.get('default')
                if tol1 is None:
                    raise TypeError('Can not find tolerance for {}'.format(dtype.type))
            return tol1
        else:
            return tol
    dtype = result.dtype
    rtol1 = _resolve(dtype, rtol)
    atol1 = _resolve(dtype, atol)
    return (rtol1, atol1)

def numpy_cupy_allclose(rtol=1e-07, atol=0, err_msg='', verbose=True, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None, contiguous_check=True, *, _check_sparse_format=True):
    if False:
        for i in range(10):
            print('nop')
    "Decorator that checks NumPy results and CuPy ones are close.\n\n    Args:\n         rtol(float or dict): Relative tolerance. Besides a float value, a\n             dictionary that maps a dtypes to a float value can be supplied to\n             adjust tolerance per dtype. If the dictionary has ``'default'``\n             string as its key, its value is used as the default tolerance in\n             case any dtype keys do not match.\n         atol(float or dict): Absolute tolerance. Besides a float value, a\n             dictionary can be supplied as ``rtol``.\n         err_msg(str): The error message to be printed in case of failure.\n         verbose(bool): If ``True``, the conflicting values are\n             appended to the error message.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         type_check(bool): If ``True``, consistency of dtype is also checked.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True`` all error types are acceptable.\n             If it is ``False`` no error is acceptable.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n         contiguous_check(bool): If ``True``, consistency of contiguity is\n             also checked.\n\n    Decorated test fixture is required to return the arrays whose values are\n    close between ``numpy`` case and ``cupy`` case.\n    For example, this test case checks ``numpy.zeros`` and ``cupy.zeros``\n    should return same value.\n\n    >>> import unittest\n    >>> from cupy import testing\n    >>> class TestFoo(unittest.TestCase):\n    ...\n    ...     @testing.numpy_cupy_allclose()\n    ...     def test_foo(self, xp):\n    ...         # ...\n    ...         # Prepare data with xp\n    ...         # ...\n    ...\n    ...         xp_result = xp.zeros(10)\n    ...         return xp_result\n\n    .. seealso:: :func:`cupy.testing.assert_allclose`\n    "
    _check_tolerance_keys(rtol, atol)
    if not type_check:
        if isinstance(rtol, dict) or isinstance(atol, dict):
            raise TypeError('When `type_check` is `False`, `rtol` and `atol` must be supplied as float.')

    def check_func(c, n):
        if False:
            return 10
        (rtol1, atol1) = _resolve_tolerance(type_check, c, rtol, atol)
        _array.assert_allclose(c, n, rtol1, atol1, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, contiguous_check, accept_error, sp_name, scipy_name, _check_sparse_format)

def numpy_cupy_array_almost_equal(decimal=6, err_msg='', verbose=True, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None):
    if False:
        for i in range(10):
            print('nop')
    'Decorator that checks NumPy results and CuPy ones are almost equal.\n\n    Args:\n         decimal(int): Desired precision.\n         err_msg(str): The error message to be printed in case of failure.\n         verbose(bool): If ``True``, the conflicting values\n             are appended to the error message.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         type_check(bool): If ``True``, consistency of dtype is also checked.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True`` all error types are acceptable.\n             If it is ``False`` no error is acceptable.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n\n    Decorated test fixture is required to return the same arrays\n    in the sense of :func:`cupy.testing.assert_array_almost_equal`\n    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.\n\n    .. seealso:: :func:`cupy.testing.assert_array_almost_equal`\n    '

    def check_func(x, y):
        if False:
            while True:
                i = 10
        _array.assert_array_almost_equal(x, y, decimal, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, False, accept_error, sp_name, scipy_name)

def numpy_cupy_array_almost_equal_nulp(nulp=1, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None):
    if False:
        while True:
            i = 10
    'Decorator that checks results of NumPy and CuPy are equal w.r.t. spacing.\n\n    Args:\n         nulp(int): The maximum number of unit in the last place for tolerance.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         type_check(bool): If ``True``, consistency of dtype is also checked.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True``, all error types are acceptable.\n             If it is ``False``, no error is acceptable.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n\n    Decorated test fixture is required to return the same arrays\n    in the sense of :func:`cupy.testing.assert_array_almost_equal_nulp`\n    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.\n\n    .. seealso:: :func:`cupy.testing.assert_array_almost_equal_nulp`\n    '

    def check_func(x, y):
        if False:
            return 10
        _array.assert_array_almost_equal_nulp(x, y, nulp)
    return _make_decorator(check_func, name, type_check, False, accept_error, sp_name, scipy_name=None)

def numpy_cupy_array_max_ulp(maxulp=1, dtype=None, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None):
    if False:
        for i in range(10):
            print('nop')
    'Decorator that checks results of NumPy and CuPy ones are equal w.r.t. ulp.\n\n    Args:\n         maxulp(int): The maximum number of units in the last place\n             that elements of resulting two arrays can differ.\n         dtype(numpy.dtype): Data-type to convert the resulting\n             two array to if given.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         type_check(bool): If ``True``, consistency of dtype is also checked.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True`` all error types are acceptable.\n             If it is ``False`` no error is acceptable.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n\n    Decorated test fixture is required to return the same arrays\n    in the sense of :func:`assert_array_max_ulp`\n    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.\n\n    .. seealso:: :func:`cupy.testing.assert_array_max_ulp`\n\n    '

    def check_func(x, y):
        if False:
            print('Hello World!')
        _array.assert_array_max_ulp(x, y, maxulp, dtype)
    return _make_decorator(check_func, name, type_check, False, accept_error, sp_name, scipy_name)

def numpy_cupy_array_equal(err_msg='', verbose=True, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None, strides_check=False):
    if False:
        while True:
            i = 10
    'Decorator that checks NumPy results and CuPy ones are equal.\n\n    Args:\n         err_msg(str): The error message to be printed in case of failure.\n         verbose(bool): If ``True``, the conflicting values are\n             appended to the error message.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         type_check(bool): If ``True``, consistency of dtype is also checked.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True`` all error types are acceptable.\n             If it is ``False`` no error is acceptable.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n         strides_check(bool): If ``True``, consistency of strides is also\n             checked.\n\n    Decorated test fixture is required to return the same arrays\n    in the sense of :func:`numpy_cupy_array_equal`\n    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.\n\n    .. seealso:: :func:`cupy.testing.assert_array_equal`\n    '

    def check_func(x, y):
        if False:
            i = 10
            return i + 15
        _array.assert_array_equal(x, y, err_msg, verbose, strides_check)
    return _make_decorator(check_func, name, type_check, False, accept_error, sp_name, scipy_name)

def numpy_cupy_array_list_equal(err_msg='', verbose=True, name='xp', sp_name=None, scipy_name=None):
    if False:
        return 10
    "Decorator that checks the resulting lists of NumPy and CuPy's one are equal.\n\n    Args:\n         err_msg(str): The error message to be printed in case of failure.\n         verbose(bool): If ``True``, the conflicting values are appended\n             to the error message.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n\n    Decorated test fixture is required to return the same list of arrays\n    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.\n\n    .. seealso:: :func:`cupy.testing.assert_array_list_equal`\n    "
    warnings.warn('numpy_cupy_array_list_equal is deprecated. Use numpy_cupy_array_equal instead.', DeprecationWarning)

    def check_func(x, y):
        if False:
            i = 10
            return i + 15
        _array.assert_array_equal(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, False, False, False, sp_name, scipy_name)

def numpy_cupy_array_less(err_msg='', verbose=True, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None):
    if False:
        print('Hello World!')
    'Decorator that checks the CuPy result is less than NumPy result.\n\n    Args:\n         err_msg(str): The error message to be printed in case of failure.\n         verbose(bool): If ``True``, the conflicting values are\n             appended to the error message.\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         type_check(bool): If ``True``, consistency of dtype is also checked.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True`` all error types are acceptable.\n             If it is ``False`` no error is acceptable.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n\n    Decorated test fixture is required to return the smaller array\n    when ``xp`` is ``cupy`` than the one when ``xp`` is ``numpy``.\n\n    .. seealso:: :func:`cupy.testing.assert_array_less`\n    '

    def check_func(x, y):
        if False:
            while True:
                i = 10
        _array.assert_array_less(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, False, accept_error, sp_name, scipy_name)

def numpy_cupy_equal(name='xp', sp_name=None, scipy_name=None):
    if False:
        while True:
            i = 10
    'Decorator that checks NumPy results are equal to CuPy ones.\n\n    Args:\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n\n    Decorated test fixture is required to return the same results\n    even if ``xp`` is ``numpy`` or ``cupy``.\n    '

    def decorator(impl):
        if False:
            while True:
                i = 10

        @_wraps_partial_xp(impl, name, sp_name, scipy_name)
        def test_func(*args, **kw):
            if False:
                i = 10
                return i + 15
            (cupy_result, cupy_error, numpy_result, numpy_error) = _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name)
            if cupy_error or numpy_error:
                _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=False)
                return
            if cupy_result != numpy_result:
                message = 'Results are not equal:\ncupy: %s\nnumpy: %s' % (str(cupy_result), str(numpy_result))
                raise AssertionError(message)
        return test_func
    return decorator

def numpy_cupy_raises(name='xp', sp_name=None, scipy_name=None, accept_error=Exception):
    if False:
        return 10
    'Decorator that checks the NumPy and CuPy throw same errors.\n\n    Args:\n         name(str): Argument name whose value is either\n             ``numpy`` or ``cupy`` module.\n         sp_name(str or None): Argument name whose value is either\n             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no\n             argument is given for the modules.\n         scipy_name(str or None): Argument name whose value is either ``scipy``\n             or ``cupyx.scipy`` module. If ``None``, no argument is given for\n             the modules.\n         accept_error(bool, Exception or tuple of Exception): Specify\n             acceptable errors. When both NumPy test and CuPy test raises the\n             same type of errors, and the type of the errors is specified with\n             this argument, the errors are ignored and not raised.\n             If it is ``True`` all error types are acceptable.\n             If it is ``False`` no error is acceptable.\n\n    Decorated test fixture is required throw same errors\n    even if ``xp`` is ``numpy`` or ``cupy``.\n    '
    warnings.warn('cupy.testing.numpy_cupy_raises is deprecated.', DeprecationWarning)

    def decorator(impl):
        if False:
            i = 10
            return i + 15

        @_wraps_partial_xp(impl, name, sp_name, scipy_name)
        def test_func(*args, **kw):
            if False:
                return 10
            (cupy_result, cupy_error, numpy_result, numpy_error) = _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name)
            _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=accept_error)
        return test_func
    return decorator

def for_dtypes(dtypes, name='dtype'):
    if False:
        return 10
    'Decorator for parameterized dtype test.\n\n    Args:\n         dtypes(list of dtypes): dtypes to be tested.\n         name(str): Argument name to which specified dtypes are passed.\n\n    This decorator adds a keyword argument specified by ``name``\n    to the test fixture. Then, it runs the fixtures in parallel\n    by passing the each element of ``dtypes`` to the named\n    argument.\n    '

    def decorator(impl):
        if False:
            i = 10
            return i + 15

        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            if False:
                print('Hello World!')
            for dtype in dtypes:
                try:
                    kw[name] = numpy.dtype(dtype).type
                    impl(*args, **kw)
                except _skip_classes as e:
                    print('skipped: {} = {} ({})'.format(name, dtype, e))
                except Exception:
                    print(name, 'is', dtype)
                    raise
        return test_func
    return decorator
_complex_dtypes = (numpy.complex64, numpy.complex128)
_regular_float_dtypes = (numpy.float64, numpy.float32)
_float_dtypes = _regular_float_dtypes + (numpy.float16,)
_signed_dtypes = tuple((numpy.dtype(i).type for i in 'bhilq'))
_unsigned_dtypes = tuple((numpy.dtype(i).type for i in 'BHILQ'))
_int_dtypes = _signed_dtypes + _unsigned_dtypes
_int_bool_dtypes = _int_dtypes + (numpy.bool_,)
_regular_dtypes = _regular_float_dtypes + _int_bool_dtypes
_dtypes = _float_dtypes + _int_bool_dtypes

def _make_all_dtypes(no_float16, no_bool, no_complex):
    if False:
        print('Hello World!')
    if no_float16:
        dtypes = _regular_float_dtypes
    else:
        dtypes = _float_dtypes
    if no_bool:
        dtypes += _int_dtypes
    else:
        dtypes += _int_bool_dtypes
    if not no_complex:
        dtypes += _complex_dtypes
    return dtypes

def for_all_dtypes(name='dtype', no_float16=False, no_bool=False, no_complex=False):
    if False:
        for i in range(10):
            print('nop')
    "Decorator that checks the fixture with all dtypes.\n\n    Args:\n         name(str): Argument name to which specified dtypes are passed.\n         no_float16(bool): If ``True``, ``numpy.float16`` is\n             omitted from candidate dtypes.\n         no_bool(bool): If ``True``, ``numpy.bool_`` is\n             omitted from candidate dtypes.\n         no_complex(bool): If ``True``, ``numpy.complex64`` and\n             ``numpy.complex128`` are omitted from candidate dtypes.\n\n    dtypes to be tested: ``numpy.complex64`` (optional),\n    ``numpy.complex128`` (optional),\n    ``numpy.float16`` (optional), ``numpy.float32``,\n    ``numpy.float64``, ``numpy.dtype('b')``, ``numpy.dtype('h')``,\n    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,\n    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,\n    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).\n\n    The usage is as follows.\n    This test fixture checks if ``cPickle`` successfully reconstructs\n    :class:`cupy.ndarray` for various dtypes.\n    ``dtype`` is an argument inserted by the decorator.\n\n    >>> import unittest\n    >>> from cupy import testing\n    >>> class TestNpz(unittest.TestCase):\n    ...\n    ...     @testing.for_all_dtypes()\n    ...     def test_pickle(self, dtype):\n    ...         a = testing.shaped_arange((2, 3, 4), dtype=dtype)\n    ...         s = pickle.dumps(a)\n    ...         b = pickle.loads(s)\n    ...         testing.assert_array_equal(a, b)\n\n    Typically, we use this decorator in combination with\n    decorators that check consistency between NumPy and CuPy like\n    :func:`cupy.testing.numpy_cupy_allclose`.\n    The following is such an example.\n\n    >>> import unittest\n    >>> from cupy import testing\n    >>> class TestMean(unittest.TestCase):\n    ...\n    ...     @testing.for_all_dtypes()\n    ...     @testing.numpy_cupy_allclose()\n    ...     def test_mean_all(self, xp, dtype):\n    ...         a = testing.shaped_arange((2, 3), xp, dtype)\n    ...         return a.mean()\n\n    .. seealso:: :func:`cupy.testing.for_dtypes`\n    "
    return for_dtypes(_make_all_dtypes(no_float16, no_bool, no_complex), name=name)

def for_float_dtypes(name='dtype', no_float16=False):
    if False:
        for i in range(10):
            print('nop')
    'Decorator that checks the fixture with float dtypes.\n\n    Args:\n         name(str): Argument name to which specified dtypes are passed.\n         no_float16(bool): If ``True``, ``numpy.float16`` is\n             omitted from candidate dtypes.\n\n    dtypes to be tested are ``numpy.float16`` (optional), ``numpy.float32``,\n    and ``numpy.float64``.\n\n    .. seealso:: :func:`cupy.testing.for_dtypes`,\n        :func:`cupy.testing.for_all_dtypes`\n    '
    if no_float16:
        return for_dtypes(_regular_float_dtypes, name=name)
    else:
        return for_dtypes(_float_dtypes, name=name)

def for_signed_dtypes(name='dtype'):
    if False:
        print('Hello World!')
    "Decorator that checks the fixture with signed dtypes.\n\n    Args:\n         name(str): Argument name to which specified dtypes are passed.\n\n    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,\n    ``numpy.dtype('i')``, ``numpy.dtype('l')``, and ``numpy.dtype('q')``.\n\n    .. seealso:: :func:`cupy.testing.for_dtypes`,\n        :func:`cupy.testing.for_all_dtypes`\n    "
    return for_dtypes(_signed_dtypes, name=name)

def for_unsigned_dtypes(name='dtype'):
    if False:
        for i in range(10):
            print('nop')
    "Decorator that checks the fixture with unsinged dtypes.\n\n    Args:\n         name(str): Argument name to which specified dtypes are passed.\n\n    dtypes to be tested are ``numpy.dtype('B')``, ``numpy.dtype('H')``,\n\n     ``numpy.dtype('I')``, ``numpy.dtype('L')``, and ``numpy.dtype('Q')``.\n\n    .. seealso:: :func:`cupy.testing.for_dtypes`,\n        :func:`cupy.testing.for_all_dtypes`\n    "
    return for_dtypes(_unsigned_dtypes, name=name)

def for_int_dtypes(name='dtype', no_bool=False):
    if False:
        return 10
    "Decorator that checks the fixture with integer and optionally bool dtypes.\n\n    Args:\n         name(str): Argument name to which specified dtypes are passed.\n         no_bool(bool): If ``True``, ``numpy.bool_`` is\n             omitted from candidate dtypes.\n\n    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,\n    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,\n    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,\n    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).\n\n    .. seealso:: :func:`cupy.testing.for_dtypes`,\n        :func:`cupy.testing.for_all_dtypes`\n    "
    if no_bool:
        return for_dtypes(_int_dtypes, name=name)
    else:
        return for_dtypes(_int_bool_dtypes, name=name)

def for_complex_dtypes(name='dtype'):
    if False:
        while True:
            i = 10
    'Decorator that checks the fixture with complex dtypes.\n\n    Args:\n         name(str): Argument name to which specified dtypes are passed.\n\n    dtypes to be tested are ``numpy.complex64`` and ``numpy.complex128``.\n\n    .. seealso:: :func:`cupy.testing.for_dtypes`,\n        :func:`cupy.testing.for_all_dtypes`\n    '
    return for_dtypes(_complex_dtypes, name=name)

def for_dtypes_combination(types, names=('dtype',), full=None):
    if False:
        print('Hello World!')
    "Decorator that checks the fixture with a product set of dtypes.\n\n    Args:\n         types(list of dtypes): dtypes to be tested.\n         names(list of str): Argument names to which dtypes are passed.\n         full(bool): If ``True``, then all combinations\n             of dtypes will be tested.\n             Otherwise, the subset of combinations will be tested\n             (see the description below).\n\n    Decorator adds the keyword arguments specified by ``names``\n    to the test fixture. Then, it runs the fixtures in parallel\n    with passing (possibly a subset of) the product set of dtypes.\n    The range of dtypes is specified by ``types``.\n\n    The combination of dtypes to be tested changes depending\n    on the option ``full``. If ``full`` is ``True``,\n    all combinations of ``types`` are tested.\n    Sometimes, such an exhaustive test can be costly.\n    So, if ``full`` is ``False``, only a subset of possible combinations\n    is randomly sampled. If ``full`` is ``None``, the behavior is\n    determined by an environment variable ``CUPY_TEST_FULL_COMBINATION``.\n    If the value is set to ``'1'``, it behaves as if ``full=True``, and\n    otherwise ``full=False``.\n    "
    types = list(types)
    if len(types) == 1:
        (name,) = names
        return for_dtypes(types, name)
    if full is None:
        full = int(os.environ.get('CUPY_TEST_FULL_COMBINATION', '0')) != 0
    if full:
        combination = _parameterized.product({name: types for name in names})
    else:
        ts = []
        for _ in range(len(names)):
            shuffled_types = types[:]
            random.shuffle(shuffled_types)
            ts.append(types + shuffled_types)
        combination = [tuple(zip(names, typs)) for typs in zip(*ts)]
        combination = [dict(assoc_list) for assoc_list in set(combination)]

    def decorator(impl):
        if False:
            i = 10
            return i + 15

        @_wraps_partial(impl, *names)
        def test_func(*args, **kw):
            if False:
                for i in range(10):
                    print('nop')
            for dtypes in combination:
                kw_copy = kw.copy()
                kw_copy.update(dtypes)
                try:
                    impl(*args, **kw_copy)
                except _skip_classes as e:
                    msg = ', '.join(('{} = {}'.format(name, dtype) for (name, dtype) in dtypes.items()))
                    print('skipped: {} ({})'.format(msg, e))
                except Exception:
                    print(dtypes)
                    raise
        return test_func
    return decorator

def for_all_dtypes_combination(names=('dtyes',), no_float16=False, no_bool=False, full=None, no_complex=False):
    if False:
        print('Hello World!')
    'Decorator that checks the fixture with a product set of all dtypes.\n\n    Args:\n         names(list of str): Argument names to which dtypes are passed.\n         no_float16(bool): If ``True``, ``numpy.float16`` is\n             omitted from candidate dtypes.\n         no_bool(bool): If ``True``, ``numpy.bool_`` is\n             omitted from candidate dtypes.\n         full(bool): If ``True``, then all combinations of dtypes\n             will be tested.\n             Otherwise, the subset of combinations will be tested\n             (see description in :func:`cupy.testing.for_dtypes_combination`).\n         no_complex(bool): If, True, ``numpy.complex64`` and\n             ``numpy.complex128`` are omitted from candidate dtypes.\n\n    .. seealso:: :func:`cupy.testing.for_dtypes_combination`\n    '
    types = _make_all_dtypes(no_float16, no_bool, no_complex)
    return for_dtypes_combination(types, names, full)

def for_signed_dtypes_combination(names=('dtype',), full=None):
    if False:
        for i in range(10):
            print('nop')
    'Decorator for parameterized test w.r.t. the product set of signed dtypes.\n\n    Args:\n         names(list of str): Argument names to which dtypes are passed.\n         full(bool): If ``True``, then all combinations of dtypes\n             will be tested.\n             Otherwise, the subset of combinations will be tested\n             (see description in :func:`cupy.testing.for_dtypes_combination`).\n\n    .. seealso:: :func:`cupy.testing.for_dtypes_combination`\n    '
    return for_dtypes_combination(_signed_dtypes, names=names, full=full)

def for_unsigned_dtypes_combination(names=('dtype',), full=None):
    if False:
        i = 10
        return i + 15
    'Decorator for parameterized test w.r.t. the product set of unsigned dtypes.\n\n    Args:\n         names(list of str): Argument names to which dtypes are passed.\n         full(bool): If ``True``, then all combinations of dtypes\n             will be tested.\n             Otherwise, the subset of combinations will be tested\n             (see description in :func:`cupy.testing.for_dtypes_combination`).\n\n    .. seealso:: :func:`cupy.testing.for_dtypes_combination`\n    '
    return for_dtypes_combination(_unsigned_dtypes, names=names, full=full)

def for_int_dtypes_combination(names=('dtype',), no_bool=False, full=None):
    if False:
        for i in range(10):
            print('nop')
    'Decorator for parameterized test w.r.t. the product set of int and boolean.\n\n    Args:\n         names(list of str): Argument names to which dtypes are passed.\n         no_bool(bool): If ``True``, ``numpy.bool_`` is\n             omitted from candidate dtypes.\n         full(bool): If ``True``, then all combinations of dtypes\n             will be tested.\n             Otherwise, the subset of combinations will be tested\n             (see description in :func:`cupy.testing.for_dtypes_combination`).\n\n    .. seealso:: :func:`cupy.testing.for_dtypes_combination`\n    '
    if no_bool:
        types = _int_dtypes
    else:
        types = _int_bool_dtypes
    return for_dtypes_combination(types, names, full)

def for_orders(orders, name='order'):
    if False:
        return 10
    'Decorator to parameterize tests with order.\n\n    Args:\n         orders(list of order): orders to be tested.\n         name(str): Argument name to which the specified order is passed.\n\n    This decorator adds a keyword argument specified by ``name``\n    to the test fixtures. Then, the fixtures run by passing each element of\n    ``orders`` to the named argument.\n\n    '

    def decorator(impl):
        if False:
            for i in range(10):
                print('nop')

        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            if False:
                print('Hello World!')
            for order in orders:
                try:
                    kw[name] = order
                    impl(*args, **kw)
                except Exception:
                    print(name, 'is', order)
                    raise
        return test_func
    return decorator

def for_CF_orders(name='order'):
    if False:
        return 10
    "Decorator that checks the fixture with orders 'C' and 'F'.\n\n    Args:\n         name(str): Argument name to which the specified order is passed.\n\n    .. seealso:: :func:`cupy.testing.for_all_dtypes`\n\n    "
    return for_orders([None, 'C', 'F', 'c', 'f'], name)

def for_contiguous_axes(name='axis'):
    if False:
        i = 10
        return i + 15
    'Decorator for parametrizing tests with possible contiguous axes.\n\n    Args:\n        name(str): Argument name to which specified axis are passed.\n\n    .. note::\n        1. Adapted from tests/cupy_tests/fft_tests/test_fft.py.\n        2. Example: for ``shape = (1, 2, 3)``, the tested axes are\n            ``[(2,), (1, 2), (0, 1, 2)]`` for the C order, and\n            ``[(0,), (0, 1), (0, 1, 2)]`` for the F order.\n    '

    def decorator(impl):
        if False:
            print('Hello World!')

        @_wraps_partial(impl, name)
        def test_func(self, *args, **kw):
            if False:
                for i in range(10):
                    print('nop')
            ndim = len(self.shape)
            order = self.order
            for i in range(ndim):
                a = ()
                if order in ('c', 'C'):
                    for j in range(ndim - 1, i - 1, -1):
                        a = (j,) + a
                elif order in ('f', 'F'):
                    for j in range(0, i + 1):
                        a = a + (j,)
                else:
                    raise ValueError('Please specify the array order.')
                try:
                    kw[name] = a
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', a, ', ndim is', ndim, ', shape is', self.shape, ', order is', order)
                    raise
        return test_func
    return decorator