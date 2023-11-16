from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import Series, isna
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays import DatetimeArray
use_bn = nanops._USE_BOTTLENECK

@pytest.fixture
def disable_bottleneck(monkeypatch):
    if False:
        i = 10
        return i + 15
    with monkeypatch.context() as m:
        m.setattr(nanops, '_USE_BOTTLENECK', False)
        yield

@pytest.fixture
def arr_shape():
    if False:
        print('Hello World!')
    return (11, 7)

@pytest.fixture
def arr_float(arr_shape):
    if False:
        i = 10
        return i + 15
    return np.random.default_rng(2).standard_normal(arr_shape)

@pytest.fixture
def arr_complex(arr_float):
    if False:
        for i in range(10):
            print('nop')
    return arr_float + arr_float * 1j

@pytest.fixture
def arr_int(arr_shape):
    if False:
        for i in range(10):
            print('nop')
    return np.random.default_rng(2).integers(-10, 10, arr_shape)

@pytest.fixture
def arr_bool(arr_shape):
    if False:
        i = 10
        return i + 15
    return np.random.default_rng(2).integers(0, 2, arr_shape) == 0

@pytest.fixture
def arr_str(arr_float):
    if False:
        return 10
    return np.abs(arr_float).astype('S')

@pytest.fixture
def arr_utf(arr_float):
    if False:
        i = 10
        return i + 15
    return np.abs(arr_float).astype('U')

@pytest.fixture
def arr_date(arr_shape):
    if False:
        for i in range(10):
            print('nop')
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype('M8[ns]')

@pytest.fixture
def arr_tdelta(arr_shape):
    if False:
        while True:
            i = 10
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype('m8[ns]')

@pytest.fixture
def arr_nan(arr_shape):
    if False:
        print('Hello World!')
    return np.tile(np.nan, arr_shape)

@pytest.fixture
def arr_float_nan(arr_float, arr_nan):
    if False:
        while True:
            i = 10
    return np.vstack([arr_float, arr_nan])

@pytest.fixture
def arr_nan_float1(arr_nan, arr_float):
    if False:
        return 10
    return np.vstack([arr_nan, arr_float])

@pytest.fixture
def arr_nan_nan(arr_nan):
    if False:
        while True:
            i = 10
    return np.vstack([arr_nan, arr_nan])

@pytest.fixture
def arr_inf(arr_float):
    if False:
        print('Hello World!')
    return arr_float * np.inf

@pytest.fixture
def arr_float_inf(arr_float, arr_inf):
    if False:
        return 10
    return np.vstack([arr_float, arr_inf])

@pytest.fixture
def arr_nan_inf(arr_nan, arr_inf):
    if False:
        while True:
            i = 10
    return np.vstack([arr_nan, arr_inf])

@pytest.fixture
def arr_float_nan_inf(arr_float, arr_nan, arr_inf):
    if False:
        i = 10
        return i + 15
    return np.vstack([arr_float, arr_nan, arr_inf])

@pytest.fixture
def arr_nan_nan_inf(arr_nan, arr_inf):
    if False:
        i = 10
        return i + 15
    return np.vstack([arr_nan, arr_nan, arr_inf])

@pytest.fixture
def arr_obj(arr_float, arr_int, arr_bool, arr_complex, arr_str, arr_utf, arr_date, arr_tdelta):
    if False:
        print('Hello World!')
    return np.vstack([arr_float.astype('O'), arr_int.astype('O'), arr_bool.astype('O'), arr_complex.astype('O'), arr_str.astype('O'), arr_utf.astype('O'), arr_date.astype('O'), arr_tdelta.astype('O')])

@pytest.fixture
def arr_nan_nanj(arr_nan):
    if False:
        i = 10
        return i + 15
    with np.errstate(invalid='ignore'):
        return arr_nan + arr_nan * 1j

@pytest.fixture
def arr_complex_nan(arr_complex, arr_nan_nanj):
    if False:
        print('Hello World!')
    with np.errstate(invalid='ignore'):
        return np.vstack([arr_complex, arr_nan_nanj])

@pytest.fixture
def arr_nan_infj(arr_inf):
    if False:
        return 10
    with np.errstate(invalid='ignore'):
        return arr_inf * 1j

@pytest.fixture
def arr_complex_nan_infj(arr_complex, arr_nan_infj):
    if False:
        while True:
            i = 10
    with np.errstate(invalid='ignore'):
        return np.vstack([arr_complex, arr_nan_infj])

@pytest.fixture
def arr_float_1d(arr_float):
    if False:
        print('Hello World!')
    return arr_float[:, 0]

@pytest.fixture
def arr_nan_1d(arr_nan):
    if False:
        for i in range(10):
            print('nop')
    return arr_nan[:, 0]

@pytest.fixture
def arr_float_nan_1d(arr_float_nan):
    if False:
        while True:
            i = 10
    return arr_float_nan[:, 0]

@pytest.fixture
def arr_float1_nan_1d(arr_float1_nan):
    if False:
        while True:
            i = 10
    return arr_float1_nan[:, 0]

@pytest.fixture
def arr_nan_float1_1d(arr_nan_float1):
    if False:
        return 10
    return arr_nan_float1[:, 0]

class TestnanopsDataFrame:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        nanops._USE_BOTTLENECK = False
        arr_shape = (11, 7)
        self.arr_float = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_float1 = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_complex = self.arr_float + self.arr_float1 * 1j
        self.arr_int = np.random.default_rng(2).integers(-10, 10, arr_shape)
        self.arr_bool = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        self.arr_str = np.abs(self.arr_float).astype('S')
        self.arr_utf = np.abs(self.arr_float).astype('U')
        self.arr_date = np.random.default_rng(2).integers(0, 20000, arr_shape).astype('M8[ns]')
        self.arr_tdelta = np.random.default_rng(2).integers(0, 20000, arr_shape).astype('m8[ns]')
        self.arr_nan = np.tile(np.nan, arr_shape)
        self.arr_float_nan = np.vstack([self.arr_float, self.arr_nan])
        self.arr_float1_nan = np.vstack([self.arr_float1, self.arr_nan])
        self.arr_nan_float1 = np.vstack([self.arr_nan, self.arr_float1])
        self.arr_nan_nan = np.vstack([self.arr_nan, self.arr_nan])
        self.arr_inf = self.arr_float * np.inf
        self.arr_float_inf = np.vstack([self.arr_float, self.arr_inf])
        self.arr_nan_inf = np.vstack([self.arr_nan, self.arr_inf])
        self.arr_float_nan_inf = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        self.arr_nan_nan_inf = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        self.arr_obj = np.vstack([self.arr_float.astype('O'), self.arr_int.astype('O'), self.arr_bool.astype('O'), self.arr_complex.astype('O'), self.arr_str.astype('O'), self.arr_utf.astype('O'), self.arr_date.astype('O'), self.arr_tdelta.astype('O')])
        with np.errstate(invalid='ignore'):
            self.arr_nan_nanj = self.arr_nan + self.arr_nan * 1j
            self.arr_complex_nan = np.vstack([self.arr_complex, self.arr_nan_nanj])
            self.arr_nan_infj = self.arr_inf * 1j
            self.arr_complex_nan_infj = np.vstack([self.arr_complex, self.arr_nan_infj])
        self.arr_float_2d = self.arr_float
        self.arr_float1_2d = self.arr_float1
        self.arr_nan_2d = self.arr_nan
        self.arr_float_nan_2d = self.arr_float_nan
        self.arr_float1_nan_2d = self.arr_float1_nan
        self.arr_nan_float1_2d = self.arr_nan_float1
        self.arr_float_1d = self.arr_float[:, 0]
        self.arr_float1_1d = self.arr_float1[:, 0]
        self.arr_nan_1d = self.arr_nan[:, 0]
        self.arr_float_nan_1d = self.arr_float_nan[:, 0]
        self.arr_float1_nan_1d = self.arr_float1_nan[:, 0]
        self.arr_nan_float1_1d = self.arr_nan_float1[:, 0]

    def teardown_method(self):
        if False:
            print('Hello World!')
        nanops._USE_BOTTLENECK = use_bn

    def check_results(self, targ, res, axis, check_dtype=True):
        if False:
            i = 10
            return i + 15
        res = getattr(res, 'asm8', res)
        if axis != 0 and hasattr(targ, 'shape') and targ.ndim and (targ.shape != res.shape):
            res = np.split(res, [targ.shape[0]], axis=0)[0]
        try:
            tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
        except AssertionError:
            if hasattr(targ, 'dtype') and targ.dtype == 'm8[ns]':
                raise
            if not hasattr(res, 'dtype') or res.dtype.kind not in ['c', 'O']:
                raise
            if res.dtype.kind == 'O':
                if targ.dtype.kind != 'O':
                    res = res.astype(targ.dtype)
                else:
                    cast_dtype = 'c16' if hasattr(np, 'complex128') else 'f8'
                    res = res.astype(cast_dtype)
                    targ = targ.astype(cast_dtype)
            elif targ.dtype.kind == 'O':
                raise
            tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
            tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)

    def check_fun_data(self, testfunc, targfunc, testarval, targarval, skipna, check_dtype=True, empty_targfunc=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for axis in list(range(targarval.ndim)) + [None]:
            targartempval = targarval if skipna else testarval
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                targ = targfunc(targartempval, axis=axis, **kwargs)
            if targartempval.dtype == object and (targfunc is np.any or targfunc is np.all):
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)
            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
            if isinstance(targ, np.complex128) and isinstance(res, float) and np.isnan(targ) and np.isnan(res):
                targ = res
            self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna:
                res = testfunc(testarval, axis=axis, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if axis is None:
                res = testfunc(testarval, skipna=skipna, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna and axis is None:
                res = testfunc(testarval, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
        if testarval.ndim <= 1:
            return
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        self.check_fun_data(testfunc, targfunc, testarval2, targarval2, skipna=skipna, check_dtype=check_dtype, empty_targfunc=empty_targfunc, **kwargs)

    def check_fun(self, testfunc, targfunc, testar, skipna, empty_targfunc=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        targar = testar
        if testar.endswith('_nan') and hasattr(self, testar[:-4]):
            targar = testar[:-4]
        testarval = getattr(self, testar)
        targarval = getattr(self, targar)
        self.check_fun_data(testfunc, targfunc, testarval, targarval, skipna=skipna, empty_targfunc=empty_targfunc, **kwargs)

    def check_funs(self, testfunc, targfunc, skipna, allow_complex=True, allow_all_nan=True, allow_date=True, allow_tdelta=True, allow_obj=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.check_fun(testfunc, targfunc, 'arr_float', skipna, **kwargs)
        self.check_fun(testfunc, targfunc, 'arr_float_nan', skipna, **kwargs)
        self.check_fun(testfunc, targfunc, 'arr_int', skipna, **kwargs)
        self.check_fun(testfunc, targfunc, 'arr_bool', skipna, **kwargs)
        objs = [self.arr_float.astype('O'), self.arr_int.astype('O'), self.arr_bool.astype('O')]
        if allow_all_nan:
            self.check_fun(testfunc, targfunc, 'arr_nan', skipna, **kwargs)
        if allow_complex:
            self.check_fun(testfunc, targfunc, 'arr_complex', skipna, **kwargs)
            self.check_fun(testfunc, targfunc, 'arr_complex_nan', skipna, **kwargs)
            if allow_all_nan:
                self.check_fun(testfunc, targfunc, 'arr_nan_nanj', skipna, **kwargs)
            objs += [self.arr_complex.astype('O')]
        if allow_date:
            targfunc(self.arr_date)
            self.check_fun(testfunc, targfunc, 'arr_date', skipna, **kwargs)
            objs += [self.arr_date.astype('O')]
        if allow_tdelta:
            try:
                targfunc(self.arr_tdelta)
            except TypeError:
                pass
            else:
                self.check_fun(testfunc, targfunc, 'arr_tdelta', skipna, **kwargs)
                objs += [self.arr_tdelta.astype('O')]
        if allow_obj:
            self.arr_obj = np.vstack(objs)
            if allow_obj == 'convert':
                targfunc = partial(self._badobj_wrap, func=targfunc, allow_complex=allow_complex)
            self.check_fun(testfunc, targfunc, 'arr_obj', skipna, **kwargs)

    def _badobj_wrap(self, value, func, allow_complex=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if value.dtype.kind == 'O':
            if allow_complex:
                value = value.astype('c16')
            else:
                value = value.astype('f8')
        return func(value, **kwargs)

    @pytest.mark.parametrize('nan_op,np_op', [(nanops.nanany, np.any), (nanops.nanall, np.all)])
    def test_nan_funcs(self, nan_op, np_op, skipna):
        if False:
            return 10
        self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)

    def test_nansum(self, skipna):
        if False:
            while True:
                i = 10
        self.check_funs(nanops.nansum, np.sum, skipna, allow_date=False, check_dtype=False, empty_targfunc=np.nansum)

    def test_nanmean(self, skipna):
        if False:
            while True:
                i = 10
        self.check_funs(nanops.nanmean, np.mean, skipna, allow_obj=False, allow_date=False)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nanmedian(self, skipna):
        if False:
            while True:
                i = 10
        self.check_funs(nanops.nanmedian, np.median, skipna, allow_complex=False, allow_date=False, allow_obj='convert')

    @pytest.mark.parametrize('ddof', range(3))
    def test_nanvar(self, ddof, skipna):
        if False:
            i = 10
            return i + 15
        self.check_funs(nanops.nanvar, np.var, skipna, allow_complex=False, allow_date=False, allow_obj='convert', ddof=ddof)

    @pytest.mark.parametrize('ddof', range(3))
    def test_nanstd(self, ddof, skipna):
        if False:
            print('Hello World!')
        self.check_funs(nanops.nanstd, np.std, skipna, allow_complex=False, allow_date=False, allow_obj='convert', ddof=ddof)

    @pytest.mark.parametrize('ddof', range(3))
    def test_nansem(self, ddof, skipna):
        if False:
            for i in range(10):
                print('nop')
        sp_stats = pytest.importorskip('scipy.stats')
        with np.errstate(invalid='ignore'):
            self.check_funs(nanops.nansem, sp_stats.sem, skipna, allow_complex=False, allow_date=False, allow_tdelta=False, allow_obj='convert', ddof=ddof)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('nan_op,np_op', [(nanops.nanmin, np.min), (nanops.nanmax, np.max)])
    def test_nanops_with_warnings(self, nan_op, np_op, skipna):
        if False:
            for i in range(10):
                print('nop')
        self.check_funs(nan_op, np_op, skipna, allow_obj=False)

    def _argminmax_wrap(self, value, axis=None, func=None):
        if False:
            return 10
        res = func(value, axis)
        nans = np.min(value, axis)
        nullnan = isna(nans)
        if res.ndim:
            res[nullnan] = -1
        elif hasattr(nullnan, 'all') and nullnan.all() or (not hasattr(nullnan, 'all') and nullnan):
            res = -1
        return res

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nanargmax(self, skipna):
        if False:
            i = 10
            return i + 15
        func = partial(self._argminmax_wrap, func=np.argmax)
        self.check_funs(nanops.nanargmax, func, skipna, allow_obj=False)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nanargmin(self, skipna):
        if False:
            i = 10
            return i + 15
        func = partial(self._argminmax_wrap, func=np.argmin)
        self.check_funs(nanops.nanargmin, func, skipna, allow_obj=False)

    def _skew_kurt_wrap(self, values, axis=None, func=None):
        if False:
            return 10
        if not isinstance(values.dtype.type, np.floating):
            values = values.astype('f8')
        result = func(values, axis=axis, bias=False)
        if isinstance(result, np.ndarray):
            result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
            return result
        elif np.max(values) == np.min(values):
            return 0.0
        return result

    def test_nanskew(self, skipna):
        if False:
            print('Hello World!')
        sp_stats = pytest.importorskip('scipy.stats')
        func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
        with np.errstate(invalid='ignore'):
            self.check_funs(nanops.nanskew, func, skipna, allow_complex=False, allow_date=False, allow_tdelta=False)

    def test_nankurt(self, skipna):
        if False:
            print('Hello World!')
        sp_stats = pytest.importorskip('scipy.stats')
        func1 = partial(sp_stats.kurtosis, fisher=True)
        func = partial(self._skew_kurt_wrap, func=func1)
        with np.errstate(invalid='ignore'):
            self.check_funs(nanops.nankurt, func, skipna, allow_complex=False, allow_date=False, allow_tdelta=False)

    def test_nanprod(self, skipna):
        if False:
            return 10
        self.check_funs(nanops.nanprod, np.prod, skipna, allow_date=False, allow_tdelta=False, empty_targfunc=np.nanprod)

    def check_nancorr_nancov_2d(self, checkfun, targ0, targ1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        res00 = checkfun(self.arr_float_2d, self.arr_float1_2d, **kwargs)
        res01 = checkfun(self.arr_float_2d, self.arr_float1_2d, min_periods=len(self.arr_float_2d) - 1, **kwargs)
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)
        res10 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, **kwargs)
        res11 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, min_periods=len(self.arr_float_2d) - 1, **kwargs)
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)
        targ2 = np.nan
        res20 = checkfun(self.arr_nan_2d, self.arr_float1_2d, **kwargs)
        res21 = checkfun(self.arr_float_2d, self.arr_nan_2d, **kwargs)
        res22 = checkfun(self.arr_nan_2d, self.arr_nan_2d, **kwargs)
        res23 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, **kwargs)
        res24 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, min_periods=len(self.arr_float_2d) - 1, **kwargs)
        res25 = checkfun(self.arr_float_2d, self.arr_float1_2d, min_periods=len(self.arr_float_2d) + 1, **kwargs)
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def check_nancorr_nancov_1d(self, checkfun, targ0, targ1, **kwargs):
        if False:
            while True:
                i = 10
        res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
        res01 = checkfun(self.arr_float_1d, self.arr_float1_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)
        res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
        res11 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)
        targ2 = np.nan
        res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
        res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
        res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
        res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
        res24 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
        res25 = checkfun(self.arr_float_1d, self.arr_float1_1d, min_periods=len(self.arr_float_1d) + 1, **kwargs)
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def test_nancorr(self):
        if False:
            print('Hello World!')
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1)
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='pearson')

    def test_nancorr_pearson(self):
        if False:
            print('Hello World!')
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='pearson')
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='pearson')

    def test_nancorr_kendall(self):
        if False:
            return 10
        sp_stats = pytest.importorskip('scipy.stats')
        targ0 = sp_stats.kendalltau(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='kendall')
        targ0 = sp_stats.kendalltau(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='kendall')

    def test_nancorr_spearman(self):
        if False:
            for i in range(10):
                print('nop')
        sp_stats = pytest.importorskip('scipy.stats')
        targ0 = sp_stats.spearmanr(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='spearman')
        targ0 = sp_stats.spearmanr(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='spearman')

    def test_invalid_method(self):
        if False:
            return 10
        pytest.importorskip('scipy')
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        msg = "Unknown method 'foo', expected one of 'kendall', 'spearman'"
        with pytest.raises(ValueError, match=msg):
            self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='foo')

    def test_nancov(self):
        if False:
            i = 10
            return i + 15
        targ0 = np.cov(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.cov(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancov, targ0, targ1)
        targ0 = np.cov(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.cov(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancov, targ0, targ1)

@pytest.mark.parametrize('arr, correct', [('arr_complex', False), ('arr_int', False), ('arr_bool', False), ('arr_str', False), ('arr_utf', False), ('arr_complex', False), ('arr_complex_nan', False), ('arr_nan_nanj', False), ('arr_nan_infj', True), ('arr_complex_nan_infj', True)])
def test_has_infs_non_float(request, arr, correct, disable_bottleneck):
    if False:
        print('Hello World!')
    val = request.getfixturevalue(arr)
    while getattr(val, 'ndim', True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0
        if not hasattr(val, 'ndim'):
            break
        val = np.take(val, 0, axis=-1)

@pytest.mark.parametrize('arr, correct', [('arr_float', False), ('arr_nan', False), ('arr_float_nan', False), ('arr_nan_nan', False), ('arr_float_inf', True), ('arr_inf', True), ('arr_nan_inf', True), ('arr_float_nan_inf', True), ('arr_nan_nan_inf', True)])
@pytest.mark.parametrize('astype', [None, 'f4', 'f2'])
def test_has_infs_floats(request, arr, correct, astype, disable_bottleneck):
    if False:
        print('Hello World!')
    val = request.getfixturevalue(arr)
    if astype is not None:
        val = val.astype(astype)
    while getattr(val, 'ndim', True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0
        if not hasattr(val, 'ndim'):
            break
        val = np.take(val, 0, axis=-1)

@pytest.mark.parametrize('fixture', ['arr_float', 'arr_complex', 'arr_int', 'arr_bool', 'arr_str', 'arr_utf'])
def test_bn_ok_dtype(fixture, request, disable_bottleneck):
    if False:
        print('Hello World!')
    obj = request.getfixturevalue(fixture)
    assert nanops._bn_ok_dtype(obj.dtype, 'test')

@pytest.mark.parametrize('fixture', ['arr_date', 'arr_tdelta', 'arr_obj'])
def test_bn_not_ok_dtype(fixture, request, disable_bottleneck):
    if False:
        while True:
            i = 10
    obj = request.getfixturevalue(fixture)
    assert not nanops._bn_ok_dtype(obj.dtype, 'test')

class TestEnsureNumeric:

    def test_numeric_values(self):
        if False:
            for i in range(10):
                print('nop')
        assert nanops._ensure_numeric(1) == 1
        assert nanops._ensure_numeric(1.1) == 1.1
        assert nanops._ensure_numeric(1 + 2j) == 1 + 2j

    def test_ndarray(self):
        if False:
            i = 10
            return i + 15
        values = np.array([1, 2, 3])
        assert np.allclose(nanops._ensure_numeric(values), values)
        o_values = values.astype(object)
        assert np.allclose(nanops._ensure_numeric(o_values), values)
        s_values = np.array(['1', '2', '3'], dtype=object)
        msg = "Could not convert \\['1' '2' '3'\\] to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)
        s_values = np.array(['foo', 'bar', 'baz'], dtype=object)
        msg = 'Could not convert .* to numeric'
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)

    def test_convertable_values(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError, match="Could not convert string '1' to numeric"):
            nanops._ensure_numeric('1')
        with pytest.raises(TypeError, match="Could not convert string '1.1' to numeric"):
            nanops._ensure_numeric('1.1')
        with pytest.raises(TypeError, match="Could not convert string '1\\+1j' to numeric"):
            nanops._ensure_numeric('1+1j')

    def test_non_convertable_values(self):
        if False:
            print('Hello World!')
        msg = "Could not convert string 'foo' to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric('foo')
        msg = 'argument must be a string or a number'
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric({})
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric([])

class TestNanvarFixedValues:

    @pytest.fixture
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        return 3.0

    @pytest.fixture
    def samples(self, variance):
        if False:
            while True:
                i = 10
        return self.prng.normal(scale=variance ** 0.5, size=100000)

    def test_nanvar_all_finite(self, samples, variance):
        if False:
            while True:
                i = 10
        actual_variance = nanops.nanvar(samples)
        tm.assert_almost_equal(actual_variance, variance, rtol=0.01)

    def test_nanvar_nans(self, samples, variance):
        if False:
            for i in range(10):
                print('nop')
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples
        actual_variance = nanops.nanvar(samples_test, skipna=True)
        tm.assert_almost_equal(actual_variance, variance, rtol=0.01)
        actual_variance = nanops.nanvar(samples_test, skipna=False)
        tm.assert_almost_equal(actual_variance, np.nan, rtol=0.01)

    def test_nanstd_nans(self, samples, variance):
        if False:
            i = 10
            return i + 15
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples
        actual_std = nanops.nanstd(samples_test, skipna=True)
        tm.assert_almost_equal(actual_std, variance ** 0.5, rtol=0.01)
        actual_std = nanops.nanvar(samples_test, skipna=False)
        tm.assert_almost_equal(actual_std, np.nan, rtol=0.01)

    def test_nanvar_axis(self, samples, variance):
        if False:
            while True:
                i = 10
        samples_unif = self.prng.uniform(size=samples.shape[0])
        samples = np.vstack([samples, samples_unif])
        actual_variance = nanops.nanvar(samples, axis=1)
        tm.assert_almost_equal(actual_variance, np.array([variance, 1.0 / 12]), rtol=0.01)

    def test_nanvar_ddof(self):
        if False:
            for i in range(10):
                print('nop')
        n = 5
        samples = self.prng.uniform(size=(10000, n + 1))
        samples[:, -1] = np.nan
        variance_0 = nanops.nanvar(samples, axis=1, skipna=True, ddof=0).mean()
        variance_1 = nanops.nanvar(samples, axis=1, skipna=True, ddof=1).mean()
        variance_2 = nanops.nanvar(samples, axis=1, skipna=True, ddof=2).mean()
        var = 1.0 / 12
        tm.assert_almost_equal(variance_1, var, rtol=0.01)
        tm.assert_almost_equal(variance_0, (n - 1.0) / n * var, rtol=0.01)
        tm.assert_almost_equal(variance_2, (n - 1.0) / (n - 2.0) * var, rtol=0.01)

    @pytest.mark.parametrize('axis', range(2))
    @pytest.mark.parametrize('ddof', range(3))
    def test_ground_truth(self, axis, ddof):
        if False:
            for i in range(10):
                print('nop')
        samples = np.empty((4, 4))
        samples[:3, :3] = np.array([[0.97303362, 0.21869576, 0.55560287], [0.72980153, 0.03109364, 0.99155171], [0.09317602, 0.60078248, 0.15871292]])
        samples[3] = samples[:, 3] = np.nan
        variance = np.array([[[0.13762259, 0.05619224, 0.11568816], [0.20643388, 0.08428837, 0.17353224], [0.41286776, 0.16857673, 0.34706449]], [[0.09519783, 0.16435395, 0.05082054], [0.14279674, 0.24653093, 0.07623082], [0.28559348, 0.49306186, 0.15246163]]])
        var = nanops.nanvar(samples, skipna=True, axis=axis, ddof=ddof)
        tm.assert_almost_equal(var[:3], variance[axis, ddof])
        assert np.isnan(var[3])
        std = nanops.nanstd(samples, skipna=True, axis=axis, ddof=ddof)
        tm.assert_almost_equal(std[:3], variance[axis, ddof] ** 0.5)
        assert np.isnan(std[3])

    @pytest.mark.parametrize('ddof', range(3))
    def test_nanstd_roundoff(self, ddof):
        if False:
            print('Hello World!')
        data = Series(766897346 * np.ones(10))
        result = data.std(ddof=ddof)
        assert result == 0.0

    @property
    def prng(self):
        if False:
            i = 10
            return i + 15
        return np.random.default_rng(2)

class TestNanskewFixedValues:

    @pytest.fixture
    def samples(self):
        if False:
            for i in range(10):
                print('nop')
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_skew(self):
        if False:
            print('Hello World!')
        return -0.1875895205961754

    @pytest.mark.parametrize('val', [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        if False:
            return 10
        data = val * np.ones(300)
        skew = nanops.nanskew(data)
        assert skew == 0.0

    def test_all_finite(self):
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta) = (0.3, 0.1)
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(left_tailed) < 0
        (alpha, beta) = (0.1, 0.3)
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(right_tailed) > 0

    def test_ground_truth(self, samples, actual_skew):
        if False:
            while True:
                i = 10
        skew = nanops.nanskew(samples)
        tm.assert_almost_equal(skew, actual_skew)

    def test_axis(self, samples, actual_skew):
        if False:
            for i in range(10):
                print('nop')
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        skew = nanops.nanskew(samples, axis=1)
        tm.assert_almost_equal(skew, np.array([actual_skew, np.nan]))

    def test_nans(self, samples):
        if False:
            while True:
                i = 10
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=False)
        assert np.isnan(skew)

    def test_nans_skipna(self, samples, actual_skew):
        if False:
            while True:
                i = 10
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=True)
        tm.assert_almost_equal(skew, actual_skew)

    @property
    def prng(self):
        if False:
            while True:
                i = 10
        return np.random.default_rng(2)

class TestNankurtFixedValues:

    @pytest.fixture
    def samples(self):
        if False:
            for i in range(10):
                print('nop')
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_kurt(self):
        if False:
            print('Hello World!')
        return -1.2058303433799713

    @pytest.mark.parametrize('val', [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        if False:
            print('Hello World!')
        data = val * np.ones(300)
        kurt = nanops.nankurt(data)
        assert kurt == 0.0

    def test_all_finite(self):
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta) = (0.3, 0.1)
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nankurt(left_tailed) < 2
        (alpha, beta) = (0.1, 0.3)
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nankurt(right_tailed) < 0

    def test_ground_truth(self, samples, actual_kurt):
        if False:
            print('Hello World!')
        kurt = nanops.nankurt(samples)
        tm.assert_almost_equal(kurt, actual_kurt)

    def test_axis(self, samples, actual_kurt):
        if False:
            while True:
                i = 10
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        kurt = nanops.nankurt(samples, axis=1)
        tm.assert_almost_equal(kurt, np.array([actual_kurt, np.nan]))

    def test_nans(self, samples):
        if False:
            print('Hello World!')
        samples = np.hstack([samples, np.nan])
        kurt = nanops.nankurt(samples, skipna=False)
        assert np.isnan(kurt)

    def test_nans_skipna(self, samples, actual_kurt):
        if False:
            i = 10
            return i + 15
        samples = np.hstack([samples, np.nan])
        kurt = nanops.nankurt(samples, skipna=True)
        tm.assert_almost_equal(kurt, actual_kurt)

    @property
    def prng(self):
        if False:
            i = 10
            return i + 15
        return np.random.default_rng(2)

class TestDatetime64NaNOps:

    @pytest.fixture(params=['s', 'ms', 'us', 'ns'])
    def unit(self, request):
        if False:
            for i in range(10):
                print('nop')
        return request.param

    def test_nanmean(self, unit):
        if False:
            while True:
                i = 10
        dti = pd.date_range('2016-01-01', periods=3).as_unit(unit)
        expected = dti[1]
        for obj in [dti, DatetimeArray(dti), Series(dti)]:
            result = nanops.nanmean(obj)
            assert result == expected
        dti2 = dti.insert(1, pd.NaT)
        for obj in [dti2, DatetimeArray(dti2), Series(dti2)]:
            result = nanops.nanmean(obj)
            assert result == expected

    @pytest.mark.parametrize('constructor', ['M8', 'm8'])
    def test_nanmean_skipna_false(self, constructor, unit):
        if False:
            for i in range(10):
                print('nop')
        dtype = f'{constructor}[{unit}]'
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(4, 3)
        arr[-1, -1] = 'NaT'
        result = nanops.nanmean(arr, skipna=False)
        assert np.isnat(result)
        assert result.dtype == dtype
        result = nanops.nanmean(arr, axis=0, skipna=False)
        expected = np.array([4, 5, 'NaT'], dtype=arr.dtype)
        tm.assert_numpy_array_equal(result, expected)
        result = nanops.nanmean(arr, axis=1, skipna=False)
        expected = np.array([arr[0, 1], arr[1, 1], arr[2, 1], arr[-1, -1]])
        tm.assert_numpy_array_equal(result, expected)

def test_use_bottleneck():
    if False:
        print('Hello World!')
    if nanops._BOTTLENECK_INSTALLED:
        with pd.option_context('use_bottleneck', True):
            assert pd.get_option('use_bottleneck')
        with pd.option_context('use_bottleneck', False):
            assert not pd.get_option('use_bottleneck')

@pytest.mark.parametrize('numpy_op, expected', [(np.sum, 10), (np.nansum, 10), (np.mean, 2.5), (np.nanmean, 2.5), (np.median, 2.5), (np.nanmedian, 2.5), (np.min, 1), (np.max, 4), (np.nanmin, 1), (np.nanmax, 4)])
def test_numpy_ops(numpy_op, expected):
    if False:
        while True:
            i = 10
    result = numpy_op(Series([1, 2, 3, 4]))
    assert result == expected

@pytest.mark.parametrize('operation', [nanops.nanany, nanops.nanall, nanops.nansum, nanops.nanmean, nanops.nanmedian, nanops.nanstd, nanops.nanvar, nanops.nansem, nanops.nanargmax, nanops.nanargmin, nanops.nanmax, nanops.nanmin, nanops.nanskew, nanops.nankurt, nanops.nanprod])
def test_nanops_independent_of_mask_param(operation):
    if False:
        for i in range(10):
            print('nop')
    ser = Series([1, 2, np.nan, 3, np.nan, 4])
    mask = ser.isna()
    median_expected = operation(ser._values)
    median_result = operation(ser._values, mask=mask)
    assert median_expected == median_result

@pytest.mark.parametrize('min_count', [-1, 0])
def test_check_below_min_count_negative_or_zero_min_count(min_count):
    if False:
        print('Hello World!')
    result = nanops.check_below_min_count((21, 37), None, min_count)
    expected_result = False
    assert result == expected_result

@pytest.mark.parametrize('mask', [None, np.array([False, False, True]), np.array([True] + 9 * [False])])
@pytest.mark.parametrize('min_count, expected_result', [(1, False), (101, True)])
def test_check_below_min_count_positive_min_count(mask, min_count, expected_result):
    if False:
        print('Hello World!')
    shape = (10, 10)
    result = nanops.check_below_min_count(shape, mask, min_count)
    assert result == expected_result

@td.skip_if_windows
@td.skip_if_32bit
@pytest.mark.parametrize('min_count, expected_result', [(1, False), (2812191852, True)])
def test_check_below_min_count_large_shape(min_count, expected_result):
    if False:
        print('Hello World!')
    shape = (2244367, 1253)
    result = nanops.check_below_min_count(shape, mask=None, min_count=min_count)
    assert result == expected_result

@pytest.mark.parametrize('func', ['nanmean', 'nansum'])
def test_check_bottleneck_disallow(any_real_numpy_dtype, func):
    if False:
        for i in range(10):
            print('nop')
    assert not nanops._bn_ok_dtype(np.dtype(any_real_numpy_dtype).type, func)

@pytest.mark.parametrize('val', [2 ** 55, -2 ** 55, 20150515061816532])
def test_nanmean_overflow(disable_bottleneck, val):
    if False:
        print('Hello World!')
    ser = Series(val, index=range(500), dtype=np.int64)
    result = ser.mean()
    np_result = ser.values.mean()
    assert result == val
    assert result == np_result
    assert result.dtype == np.float64

@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.int64, np.float32, np.float64, getattr(np, 'float128', None)])
@pytest.mark.parametrize('method', ['mean', 'std', 'var', 'skew', 'kurt', 'min', 'max'])
def test_returned_dtype(disable_bottleneck, dtype, method):
    if False:
        while True:
            i = 10
    if dtype is None:
        pytest.skip('np.float128 not available')
    ser = Series(range(10), dtype=dtype)
    result = getattr(ser, method)()
    if is_integer_dtype(dtype) and method not in ['min', 'max']:
        assert result.dtype == np.float64
    else:
        assert result.dtype == dtype