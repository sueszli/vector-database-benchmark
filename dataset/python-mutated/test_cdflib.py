"""
Test cdflib functions versus mpmath, if available.

The following functions still need tests:

- ncfdtr
- ncfdtri
- ncfdtridfn
- ncfdtridfd
- ncfdtrinc
- nbdtrik
- nbdtrin
- nrdtrimn
- nrdtrisd
- pdtrik
- nctdtr
- nctdtrit
- nctdtridf
- nctdtrinc

"""
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import MissingModule, check_version, FuncData
from scipy.special._mptestutils import Arg, IntArg, get_args, mpf2float, assert_mpmath_equal
try:
    import mpmath
except ImportError:
    mpmath = MissingModule('mpmath')

class ProbArg:
    """Generate a set of probabilities on [0, 1]."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = 0
        self.b = 1

    def values(self, n):
        if False:
            return 10
        'Return an array containing approximatively n numbers.'
        m = max(1, n // 3)
        v1 = np.logspace(-30, np.log10(0.3), m)
        v2 = np.linspace(0.3, 0.7, m + 1, endpoint=False)[1:]
        v3 = 1 - np.logspace(np.log10(0.3), -15, m)
        v = np.r_[v1, v2, v3]
        return np.unique(v)

class EndpointFilter:

    def __init__(self, a, b, rtol, atol):
        if False:
            i = 10
            return i + 15
        self.a = a
        self.b = b
        self.rtol = rtol
        self.atol = atol

    def __call__(self, x):
        if False:
            return 10
        mask1 = np.abs(x - self.a) < self.rtol * np.abs(self.a) + self.atol
        mask2 = np.abs(x - self.b) < self.rtol * np.abs(self.b) + self.atol
        return np.where(mask1 | mask2, False, True)

class _CDFData:

    def __init__(self, spfunc, mpfunc, index, argspec, spfunc_first=True, dps=20, n=5000, rtol=None, atol=None, endpt_rtol=None, endpt_atol=None):
        if False:
            print('Hello World!')
        self.spfunc = spfunc
        self.mpfunc = mpfunc
        self.index = index
        self.argspec = argspec
        self.spfunc_first = spfunc_first
        self.dps = dps
        self.n = n
        self.rtol = rtol
        self.atol = atol
        if not isinstance(argspec, list):
            self.endpt_rtol = None
            self.endpt_atol = None
        elif endpt_rtol is not None or endpt_atol is not None:
            if isinstance(endpt_rtol, list):
                self.endpt_rtol = endpt_rtol
            else:
                self.endpt_rtol = [endpt_rtol] * len(self.argspec)
            if isinstance(endpt_atol, list):
                self.endpt_atol = endpt_atol
            else:
                self.endpt_atol = [endpt_atol] * len(self.argspec)
        else:
            self.endpt_rtol = None
            self.endpt_atol = None

    def idmap(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if self.spfunc_first:
            res = self.spfunc(*args)
            if np.isnan(res):
                return np.nan
            args = list(args)
            args[self.index] = res
            with mpmath.workdps(self.dps):
                res = self.mpfunc(*tuple(args))
                res = mpf2float(res.real)
        else:
            with mpmath.workdps(self.dps):
                res = self.mpfunc(*args)
                res = mpf2float(res.real)
            args = list(args)
            args[self.index] = res
            res = self.spfunc(*tuple(args))
        return res

    def get_param_filter(self):
        if False:
            print('Hello World!')
        if self.endpt_rtol is None and self.endpt_atol is None:
            return None
        filters = []
        for (rtol, atol, spec) in zip(self.endpt_rtol, self.endpt_atol, self.argspec):
            if rtol is None and atol is None:
                filters.append(None)
                continue
            elif rtol is None:
                rtol = 0.0
            elif atol is None:
                atol = 0.0
            filters.append(EndpointFilter(spec.a, spec.b, rtol, atol))
        return filters

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        args = get_args(self.argspec, self.n)
        param_filter = self.get_param_filter()
        param_columns = tuple(range(args.shape[1]))
        result_columns = args.shape[1]
        args = np.hstack((args, args[:, self.index].reshape(args.shape[0], 1)))
        FuncData(self.idmap, args, param_columns=param_columns, result_columns=result_columns, rtol=self.rtol, atol=self.atol, vectorized=False, param_filter=param_filter).check()

def _assert_inverts(*a, **kw):
    if False:
        i = 10
        return i + 15
    d = _CDFData(*a, **kw)
    d.check()

def _binomial_cdf(k, n, p):
    if False:
        i = 10
        return i + 15
    (k, n, p) = (mpmath.mpf(k), mpmath.mpf(n), mpmath.mpf(p))
    if k <= 0:
        return mpmath.mpf(0)
    elif k >= n:
        return mpmath.mpf(1)
    onemp = mpmath.fsub(1, p, exact=True)
    return mpmath.betainc(n - k, k + 1, x2=onemp, regularized=True)

def _f_cdf(dfn, dfd, x):
    if False:
        i = 10
        return i + 15
    if x < 0:
        return mpmath.mpf(0)
    (dfn, dfd, x) = (mpmath.mpf(dfn), mpmath.mpf(dfd), mpmath.mpf(x))
    ub = dfn * x / (dfn * x + dfd)
    res = mpmath.betainc(dfn / 2, dfd / 2, x2=ub, regularized=True)
    return res

def _student_t_cdf(df, t, dps=None):
    if False:
        while True:
            i = 10
    if dps is None:
        dps = mpmath.mp.dps
    with mpmath.workdps(dps):
        (df, t) = (mpmath.mpf(df), mpmath.mpf(t))
        fac = mpmath.hyp2f1(0.5, 0.5 * (df + 1), 1.5, -t ** 2 / df)
        fac *= t * mpmath.gamma(0.5 * (df + 1))
        fac /= mpmath.sqrt(mpmath.pi * df) * mpmath.gamma(0.5 * df)
        return 0.5 + fac

def _noncentral_chi_pdf(t, df, nc):
    if False:
        return 10
    res = mpmath.besseli(df / 2 - 1, mpmath.sqrt(nc * t))
    res *= mpmath.exp(-(t + nc) / 2) * (t / nc) ** (df / 4 - 1 / 2) / 2
    return res

def _noncentral_chi_cdf(x, df, nc, dps=None):
    if False:
        return 10
    if dps is None:
        dps = mpmath.mp.dps
    (x, df, nc) = (mpmath.mpf(x), mpmath.mpf(df), mpmath.mpf(nc))
    with mpmath.workdps(dps):
        res = mpmath.quad(lambda t: _noncentral_chi_pdf(t, df, nc), [0, x])
        return res

def _tukey_lmbda_quantile(p, lmbda):
    if False:
        print('Hello World!')
    return (p ** lmbda - (1 - p) ** lmbda) / lmbda

@pytest.mark.slow
@check_version(mpmath, '0.19')
class TestCDFlib:

    @pytest.mark.xfail(run=False)
    def test_bdtrik(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_inverts(sp.bdtrik, _binomial_cdf, 0, [ProbArg(), IntArg(1, 1000), ProbArg()], rtol=0.0001)

    def test_bdtrin(self):
        if False:
            return 10
        _assert_inverts(sp.bdtrin, _binomial_cdf, 1, [IntArg(1, 1000), ProbArg(), ProbArg()], rtol=0.0001, endpt_atol=[None, None, 1e-06])

    def test_btdtria(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_inverts(sp.btdtria, lambda a, b, x: mpmath.betainc(a, b, x2=x, regularized=True), 0, [ProbArg(), Arg(0, 100.0, inclusive_a=False), Arg(0, 1, inclusive_a=False, inclusive_b=False)], rtol=1e-06)

    def test_btdtrib(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_inverts(sp.btdtrib, lambda a, b, x: mpmath.betainc(a, b, x2=x, regularized=True), 1, [Arg(0, 100.0, inclusive_a=False), ProbArg(), Arg(0, 1, inclusive_a=False, inclusive_b=False)], rtol=1e-07, endpt_atol=[None, 1e-18, 1e-15])

    @pytest.mark.xfail(run=False)
    def test_fdtridfd(self):
        if False:
            i = 10
            return i + 15
        _assert_inverts(sp.fdtridfd, _f_cdf, 1, [IntArg(1, 100), ProbArg(), Arg(0, 100, inclusive_a=False)], rtol=1e-07)

    def test_gdtria(self):
        if False:
            print('Hello World!')
        _assert_inverts(sp.gdtria, lambda a, b, x: mpmath.gammainc(b, b=a * x, regularized=True), 0, [ProbArg(), Arg(0, 1000.0, inclusive_a=False), Arg(0, 10000.0, inclusive_a=False)], rtol=1e-07, endpt_atol=[None, 1e-07, 1e-10])

    def test_gdtrib(self):
        if False:
            while True:
                i = 10
        _assert_inverts(sp.gdtrib, lambda a, b, x: mpmath.gammainc(b, b=a * x, regularized=True), 1, [Arg(0, 100.0, inclusive_a=False), ProbArg(), Arg(0, 1000.0, inclusive_a=False)], rtol=1e-05)

    def test_gdtrix(self):
        if False:
            while True:
                i = 10
        _assert_inverts(sp.gdtrix, lambda a, b, x: mpmath.gammainc(b, b=a * x, regularized=True), 2, [Arg(0, 1000.0, inclusive_a=False), Arg(0, 1000.0, inclusive_a=False), ProbArg()], rtol=1e-07, endpt_atol=[None, 1e-07, 1e-10])

    def test_stdtr(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sp.stdtr, _student_t_cdf, [IntArg(1, 100), Arg(1e-10, np.inf)], rtol=1e-07)

    @pytest.mark.xfail(run=False)
    def test_stdtridf(self):
        if False:
            while True:
                i = 10
        _assert_inverts(sp.stdtridf, _student_t_cdf, 0, [ProbArg(), Arg()], rtol=1e-07)

    def test_stdtrit(self):
        if False:
            i = 10
            return i + 15
        _assert_inverts(sp.stdtrit, _student_t_cdf, 1, [IntArg(1, 100), ProbArg()], rtol=1e-07, endpt_atol=[None, 1e-10])

    def test_chdtriv(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_inverts(sp.chdtriv, lambda v, x: mpmath.gammainc(v / 2, b=x / 2, regularized=True), 0, [ProbArg(), IntArg(1, 100)], rtol=0.0001)

    @pytest.mark.xfail(run=False)
    def test_chndtridf(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_inverts(sp.chndtridf, _noncentral_chi_cdf, 1, [Arg(0, 100, inclusive_a=False), ProbArg(), Arg(0, 100, inclusive_a=False)], n=1000, rtol=0.0001, atol=1e-15)

    @pytest.mark.xfail(run=False)
    def test_chndtrinc(self):
        if False:
            print('Hello World!')
        _assert_inverts(sp.chndtrinc, _noncentral_chi_cdf, 2, [Arg(0, 100, inclusive_a=False), IntArg(1, 100), ProbArg()], n=1000, rtol=0.0001, atol=1e-15)

    def test_chndtrix(self):
        if False:
            return 10
        _assert_inverts(sp.chndtrix, _noncentral_chi_cdf, 0, [ProbArg(), IntArg(1, 100), Arg(0, 100, inclusive_a=False)], n=1000, rtol=0.0001, atol=1e-15, endpt_atol=[1e-06, None, None])

    def test_tklmbda_zero_shape(self):
        if False:
            print('Hello World!')
        one = mpmath.mpf(1)
        assert_mpmath_equal(lambda x: sp.tklmbda(x, 0), lambda x: one / (mpmath.exp(-x) + one), [Arg()], rtol=1e-07)

    def test_tklmbda_neg_shape(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_inverts(sp.tklmbda, _tukey_lmbda_quantile, 0, [ProbArg(), Arg(-25, 0, inclusive_b=False)], spfunc_first=False, rtol=1e-05, endpt_atol=[1e-09, 1e-05])

    @pytest.mark.xfail(run=False)
    def test_tklmbda_pos_shape(self):
        if False:
            return 10
        _assert_inverts(sp.tklmbda, _tukey_lmbda_quantile, 0, [ProbArg(), Arg(0, 100, inclusive_a=False)], spfunc_first=False, rtol=1e-05)

    @pytest.mark.parametrize('lmbda', [0.5, 1.0, 8.0])
    def test_tklmbda_lmbda1(self, lmbda):
        if False:
            print('Hello World!')
        bound = 1 / lmbda
        assert_equal(sp.tklmbda([-bound, bound], lmbda), [0.0, 1.0])

def test_nonfinite():
    if False:
        print('Hello World!')
    funcs = [('btdtria', 3), ('btdtrib', 3), ('bdtrik', 3), ('bdtrin', 3), ('chdtriv', 2), ('chndtr', 3), ('chndtrix', 3), ('chndtridf', 3), ('chndtrinc', 3), ('fdtridfd', 3), ('ncfdtr', 4), ('ncfdtri', 4), ('ncfdtridfn', 4), ('ncfdtridfd', 4), ('ncfdtrinc', 4), ('gdtrix', 3), ('gdtrib', 3), ('gdtria', 3), ('nbdtrik', 3), ('nbdtrin', 3), ('nrdtrimn', 3), ('nrdtrisd', 3), ('pdtrik', 2), ('stdtr', 2), ('stdtrit', 2), ('stdtridf', 2), ('nctdtr', 3), ('nctdtrit', 3), ('nctdtridf', 3), ('nctdtrinc', 3), ('tklmbda', 2)]
    np.random.seed(1)
    for (func, numargs) in funcs:
        func = getattr(sp, func)
        args_choices = [(float(x), np.nan, np.inf, -np.inf) for x in np.random.rand(numargs)]
        for args in itertools.product(*args_choices):
            res = func(*args)
            if any((np.isnan(x) for x in args)):
                assert_equal(res, np.nan)
            else:
                pass

def test_chndtrix_gh2158():
    if False:
        i = 10
        return i + 15
    res = sp.chndtrix(0.999999, 2, np.arange(20.0) + 1e-06)
    res_exp = [27.63103493142305, 35.2572858995054, 39.97396073236288, 43.88033702110538, 47.35206403482798, 50.54112500166103, 53.52720257322766, 56.3583004286781, 59.06600769498512, 61.67243118946381, 64.19376191277179, 66.64228141346548, 69.0275692720018, 71.35726934749408, 73.63759723904816, 75.87368842650227, 78.0698443118572, 80.22971052389806, 82.35640899964173, 84.45263768373256]
    assert_allclose(res, res_exp)