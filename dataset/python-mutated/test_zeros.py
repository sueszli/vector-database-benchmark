import pytest
from functools import lru_cache
from numpy.testing import assert_warns, assert_, assert_allclose, assert_equal, assert_array_equal, assert_array_less, suppress_warnings
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import _zeros_py as zeros, newton, root_scalar, OptimizeResult
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
TOL = 4 * np.finfo(float).eps
_FLOAT_EPS = finfo(float).eps
bracket_methods = [zeros.bisect, zeros.ridder, zeros.brentq, zeros.brenth, zeros.toms748]
gradient_methods = [zeros.newton]
all_methods = bracket_methods + gradient_methods

def f1(x):
    if False:
        while True:
            i = 10
    return x ** 2 - 2 * x - 1

def f1_1(x):
    if False:
        while True:
            i = 10
    return 2 * x - 2

def f1_2(x):
    if False:
        print('Hello World!')
    return 2.0 + 0 * x

def f1_and_p_and_pp(x):
    if False:
        while True:
            i = 10
    return (f1(x), f1_1(x), f1_2(x))

def f2(x):
    if False:
        while True:
            i = 10
    return exp(x) - cos(x)

def f2_1(x):
    if False:
        for i in range(10):
            print('nop')
    return exp(x) + sin(x)

def f2_2(x):
    if False:
        while True:
            i = 10
    return exp(x) + cos(x)

@lru_cache
def f_lrucached(x):
    if False:
        return 10
    return x

class TestScalarRootFinders:
    xtol = 4 * np.finfo(float).eps
    rtol = 4 * np.finfo(float).eps

    def _run_one_test(self, tc, method, sig_args_keys=None, sig_kwargs_keys=None, **kwargs):
        if False:
            print('Hello World!')
        method_args = []
        for k in sig_args_keys or []:
            if k not in tc:
                k = {'a': 'x0', 'b': 'x1', 'func': 'f'}.get(k, k)
            method_args.append(tc[k])
        method_kwargs = dict(**kwargs)
        method_kwargs.update({'full_output': True, 'disp': False})
        for k in sig_kwargs_keys or []:
            method_kwargs[k] = tc[k]
        root = tc.get('root')
        func_args = tc.get('args', ())
        try:
            (r, rr) = method(*method_args, args=func_args, **method_kwargs)
            return (root, rr, tc)
        except Exception:
            return (root, zeros.RootResults(nan, -1, -1, zeros._EVALUEERR, method), tc)

    def run_tests(self, tests, method, name, known_fail=None, **kwargs):
        if False:
            print('Hello World!')
        "Run test-cases using the specified method and the supplied signature.\n\n        Extract the arguments for the method call from the test case\n        dictionary using the supplied keys for the method's signature."
        sig = _getfullargspec(method)
        assert_(not sig.kwonlyargs)
        nDefaults = len(sig.defaults)
        nRequired = len(sig.args) - nDefaults
        sig_args_keys = sig.args[:nRequired]
        sig_kwargs_keys = []
        if name in ['secant', 'newton', 'halley']:
            if name in ['newton', 'halley']:
                sig_kwargs_keys.append('fprime')
                if name in ['halley']:
                    sig_kwargs_keys.append('fprime2')
            kwargs['tol'] = self.xtol
        else:
            kwargs['xtol'] = self.xtol
            kwargs['rtol'] = self.rtol
        results = [list(self._run_one_test(tc, method, sig_args_keys=sig_args_keys, sig_kwargs_keys=sig_kwargs_keys, **kwargs)) for tc in tests]
        known_fail = known_fail or []
        notcvgd = [elt for elt in results if not elt[1].converged]
        notcvgd = [elt for elt in notcvgd if elt[-1]['ID'] not in known_fail]
        notcvged_IDS = [elt[-1]['ID'] for elt in notcvgd]
        assert_equal([len(notcvged_IDS), notcvged_IDS], [0, []])
        tols = {'xtol': self.xtol, 'rtol': self.rtol}
        tols.update(**kwargs)
        rtol = tols['rtol']
        atol = tols.get('tol', tols['xtol'])
        cvgd = [elt for elt in results if elt[1].converged]
        approx = [elt[1].root for elt in cvgd]
        correct = [elt[0] for elt in cvgd]
        notclose = [[a] + elt for (a, c, elt) in zip(approx, correct, cvgd) if not isclose(a, c, rtol=rtol, atol=atol) and elt[-1]['ID'] not in known_fail]
        fvs = [tc['f'](aroot, *tc.get('args', tuple())) for (aroot, c, fullout, tc) in notclose]
        notclose = [[fv] + elt for (fv, elt) in zip(fvs, notclose) if fv != 0]
        assert_equal([notclose, len(notclose)], [[], 0])
        method_from_result = [result[1].method for result in results]
        expected_method = [name for _ in results]
        assert_equal(method_from_result, expected_method)

    def run_collection(self, collection, method, name, smoothness=None, known_fail=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Run a collection of tests using the specified method.\n\n        The name is used to determine some optional arguments.'
        tests = get_tests(collection, smoothness=smoothness)
        self.run_tests(tests, method, name, known_fail=known_fail, **kwargs)

class TestBracketMethods(TestScalarRootFinders):

    @pytest.mark.parametrize('method', bracket_methods)
    @pytest.mark.parametrize('function', tstutils_functions)
    def test_basic_root_scalar(self, method, function):
        if False:
            i = 10
            return i + 15
        (a, b) = (0.5, sqrt(3))
        r = root_scalar(function, method=method.__name__, bracket=[a, b], x0=a, xtol=self.xtol, rtol=self.rtol)
        assert r.converged
        assert_allclose(r.root, 1.0, atol=self.xtol, rtol=self.rtol)
        assert r.method == method.__name__

    @pytest.mark.parametrize('method', bracket_methods)
    @pytest.mark.parametrize('function', tstutils_functions)
    def test_basic_individual(self, method, function):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (0.5, sqrt(3))
        (root, r) = method(function, a, b, xtol=self.xtol, rtol=self.rtol, full_output=True)
        assert r.converged
        assert_allclose(root, 1.0, atol=self.xtol, rtol=self.rtol)

    @pytest.mark.parametrize('method', bracket_methods)
    def test_aps_collection(self, method):
        if False:
            i = 10
            return i + 15
        self.run_collection('aps', method, method.__name__, smoothness=1)

    @pytest.mark.parametrize('method', [zeros.bisect, zeros.ridder, zeros.toms748])
    def test_chandrupatla_collection(self, method):
        if False:
            i = 10
            return i + 15
        known_fail = {'fun7.4'} if method == zeros.ridder else {}
        self.run_collection('chandrupatla', method, method.__name__, known_fail=known_fail)

    @pytest.mark.parametrize('method', bracket_methods)
    def test_lru_cached_individual(self, method):
        if False:
            return 10
        (a, b) = (-1, 1)
        (root, r) = method(f_lrucached, a, b, full_output=True)
        assert r.converged
        assert_allclose(root, 0)

class TestChandrupatla(TestScalarRootFinders):

    def f(self, q, p):
        if False:
            i = 10
            return i + 15
        return stats.norm.cdf(q) - p

    @pytest.mark.parametrize('p', [0.6, np.linspace(-0.05, 1.05, 10)])
    def test_basic(self, p):
        if False:
            for i in range(10):
                print('nop')
        res = zeros._chandrupatla(self.f, -5, 5, args=(p,))
        ref = stats.norm().ppf(p)
        np.testing.assert_allclose(res.x, ref)
        assert res.x.shape == ref.shape

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        if False:
            for i in range(10):
                print('nop')
        p = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        args = (p,)

        @np.vectorize
        def chandrupatla_single(p):
            if False:
                while True:
                    i = 10
            return zeros._chandrupatla(self.f, -5, 5, args=(p,))

        def f(*args, **kwargs):
            if False:
                print('Hello World!')
            f.f_evals += 1
            return self.f(*args, **kwargs)
        f.f_evals = 0
        res = zeros._chandrupatla(f, -5, 5, args=args)
        refs = chandrupatla_single(p).ravel()
        ref_x = [ref.x for ref in refs]
        assert_allclose(res.x.ravel(), ref_x)
        assert_equal(res.x.shape, shape)
        ref_fun = [ref.fun for ref in refs]
        assert_allclose(res.fun.ravel(), ref_fun)
        assert_equal(res.fun.shape, shape)
        assert_equal(res.fun, self.f(res.x, *args))
        ref_success = [ref.success for ref in refs]
        assert_equal(res.success.ravel(), ref_success)
        assert_equal(res.success.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        ref_flag = [ref.status for ref in refs]
        assert_equal(res.status.ravel(), ref_flag)
        assert_equal(res.status.shape, shape)
        assert np.issubdtype(res.status.dtype, np.integer)
        ref_nfev = [ref.nfev for ref in refs]
        assert_equal(res.nfev.ravel(), ref_nfev)
        assert_equal(np.max(res.nfev), f.f_evals)
        assert_equal(res.nfev.shape, res.fun.shape)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        ref_nit = [ref.nit for ref in refs]
        assert_equal(res.nit.ravel(), ref_nit)
        assert_equal(np.max(res.nit), f.f_evals - 2)
        assert_equal(res.nit.shape, res.fun.shape)
        assert np.issubdtype(res.nit.dtype, np.integer)
        ref_xl = [ref.xl for ref in refs]
        assert_allclose(res.xl.ravel(), ref_xl)
        assert_equal(res.xl.shape, shape)
        ref_xr = [ref.xr for ref in refs]
        assert_allclose(res.xr.ravel(), ref_xr)
        assert_equal(res.xr.shape, shape)
        assert_array_less(res.xl, res.xr)
        finite = np.isfinite(res.x)
        assert np.all((res.x[finite] == res.xl[finite]) | (res.x[finite] == res.xr[finite]))
        ref_fl = [ref.fl for ref in refs]
        assert_allclose(res.fl.ravel(), ref_fl)
        assert_equal(res.fl.shape, shape)
        assert_allclose(res.fl, self.f(res.xl, *args))
        ref_fr = [ref.fr for ref in refs]
        assert_allclose(res.fr.ravel(), ref_fr)
        assert_equal(res.fr.shape, shape)
        assert_allclose(res.fr, self.f(res.xr, *args))
        assert np.all(np.abs(res.fun[finite]) == np.minimum(np.abs(res.fl[finite]), np.abs(res.fr[finite])))

    def test_flags(self):
        if False:
            i = 10
            return i + 15

        def f(xs, js):
            if False:
                while True:
                    i = 10
            funcs = [lambda x: x - 2.5, lambda x: x - 10, lambda x: (x - 0.1) ** 3, lambda x: np.nan]
            return [funcs[j](x) for (x, j) in zip(xs, js)]
        args = (np.arange(4, dtype=np.int64),)
        res = zeros._chandrupatla(f, [0] * 4, [np.pi] * 4, args=args, maxiter=2)
        ref_flags = np.array([zeros._ECONVERGED, zeros._ESIGNERR, zeros._ECONVERR, zeros._EVALUEERR])
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(2585255913088665241)
        p = rng.random(size=3)
        bracket = (-5, 5)
        args = (p,)
        kwargs0 = dict(args=args, xatol=0, xrtol=0, fatol=0, frtol=0)
        kwargs = kwargs0.copy()
        kwargs['xatol'] = 0.001
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res1.xr - res1.xl, 0.001)
        kwargs['xatol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res2.xr - res2.xl, 1e-06)
        assert_array_less(res2.xr - res2.xl, res1.xr - res1.xl)
        kwargs = kwargs0.copy()
        kwargs['xrtol'] = 0.001
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res1.xr - res1.xl, 0.001 * np.abs(res1.x))
        kwargs['xrtol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res2.xr - res2.xl, 1e-06 * np.abs(res2.x))
        assert_array_less(res2.xr - res2.xl, res1.xr - res1.xl)
        kwargs = kwargs0.copy()
        kwargs['fatol'] = 0.001
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res1.fun), 0.001)
        kwargs['fatol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res2.fun), 1e-06)
        assert_array_less(np.abs(res2.fun), np.abs(res1.fun))
        kwargs = kwargs0.copy()
        kwargs['frtol'] = 0.001
        (x1, x2) = bracket
        f0 = np.minimum(abs(self.f(x1, *args)), abs(self.f(x2, *args)))
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res1.fun), 0.001 * f0)
        kwargs['frtol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res2.fun), 1e-06 * f0)
        assert_array_less(np.abs(res2.fun), np.abs(res1.fun))

    def test_maxiter_callback(self):
        if False:
            for i in range(10):
                print('nop')
        p = 0.612814
        bracket = (-5, 5)
        maxiter = 5

        def f(q, p):
            if False:
                i = 10
                return i + 15
            res = stats.norm().cdf(q) - p
            f.x = q
            f.fun = res
            return res
        f.x = None
        f.fun = None
        res = zeros._chandrupatla(f, *bracket, args=(p,), maxiter=maxiter)
        assert not np.any(res.success)
        assert np.all(res.nfev == maxiter + 2)
        assert np.all(res.nit == maxiter)

        def callback(res):
            if False:
                while True:
                    i = 10
            callback.iter += 1
            callback.res = res
            assert hasattr(res, 'x')
            if callback.iter == 0:
                assert (res.xl, res.xr) == bracket
            else:
                changed = (res.xl == callback.xl) & (res.xr != callback.xr) | (res.xl != callback.xl) & (res.xr == callback.xr)
                assert np.all(changed)
            callback.xl = res.xl
            callback.xr = res.xr
            assert res.status == zeros._EINPROGRESS
            assert_equal(self.f(res.xl, p), res.fl)
            assert_equal(self.f(res.xr, p), res.fr)
            assert_equal(self.f(res.x, p), res.fun)
            if callback.iter == maxiter:
                raise StopIteration
        callback.iter = -1
        callback.res = None
        callback.xl = None
        callback.xr = None
        res2 = zeros._chandrupatla(f, *bracket, args=(p,), callback=callback)
        for key in res.keys():
            if key == 'status':
                assert res[key] == zeros._ECONVERR
                assert callback.res[key] == zeros._EINPROGRESS
                assert res2[key] == zeros._ECALLBACK
            else:
                assert res2[key] == callback.res[key] == res[key]

    @pytest.mark.parametrize('case', optimize._tstutils._CHANDRUPATLA_TESTS)
    def test_nit_expected(self, case):
        if False:
            print('Hello World!')
        (f, bracket, root, nfeval, id) = case
        res = zeros._chandrupatla(f, *bracket, xrtol=4e-10, xatol=1e-05)
        assert_allclose(res.fun, f(root), rtol=1e-08, atol=0.002)
        assert_equal(res.nfev, nfeval)

    @pytest.mark.parametrize('root', (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize('dtype', (np.float16, np.float32, np.float64))
    def test_dtype(self, root, dtype):
        if False:
            for i in range(10):
                print('nop')
        root = dtype(root)

        def f(x, root):
            if False:
                for i in range(10):
                    print('nop')
            return ((x - root) ** 3).astype(dtype)
        res = zeros._chandrupatla(f, dtype(-3), dtype(5), args=(root,), xatol=0.001)
        assert res.x.dtype == dtype
        assert np.allclose(res.x, root, atol=0.001) or np.all(res.fun == 0)

    def test_input_validation(self):
        if False:
            print('Hello World!')
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(None, -4, 4)
        message = 'Abscissae and function output must be real numbers.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4 + 1j, 4)
        message = 'shape mismatch: objects cannot be broadcast'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, [-2, -3], [3, 4, 5])
        message = 'The shape of the array returned by `func`...'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: [x[0], x[1], x[1]], [-3, -3], [5, 5])
        message = 'Tolerances must be non-negative scalars.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, xatol=-1)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, xrtol=np.nan)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, fatol='ekki')
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, frtol=np.nan)
        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=-1)
        message = '`callback` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, callback='shrubbery')

    def test_special_cases(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99 - 1
        res = zeros._chandrupatla(f, -7, 5)
        assert res.success
        assert_allclose(res.x, 1)

        def f(x):
            if False:
                while True:
                    i = 10
            return x ** 2 - 1
        res = zeros._chandrupatla(f, 1, 1)
        assert res.success
        assert_equal(res.x, 1)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return 1 / x
        with np.errstate(invalid='ignore'):
            res = zeros._chandrupatla(f, np.inf, np.inf)
        assert res.success
        assert_equal(res.x, np.inf)

        def f(x):
            if False:
                return 10
            return x ** 3 - 1
        bracket = (-3, 5)
        res = zeros._chandrupatla(f, *bracket, maxiter=0)
        assert res.xl, res.xr == bracket
        assert res.nit == 0
        assert res.nfev == 2
        assert res.status == -2
        assert res.x == -3
        res = zeros._chandrupatla(f, *bracket, maxiter=1)
        assert res.success
        assert res.status == 0
        assert res.nit == 1
        assert res.nfev == 3
        assert_allclose(res.x, 1)

        def f(x, c):
            if False:
                while True:
                    i = 10
            return c * x - 1
        res = zeros._chandrupatla(f, -1, 1, args=3)
        assert_allclose(res.x, 1 / 3)

class TestNewton(TestScalarRootFinders):

    def test_newton_collections(self):
        if False:
            while True:
                i = 10
        known_fail = ['aps.13.00']
        known_fail += ['aps.12.05', 'aps.12.17']
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'newton', smoothness=2, known_fail=known_fail)

    def test_halley_collections(self):
        if False:
            i = 10
            return i + 15
        known_fail = ['aps.12.06', 'aps.12.07', 'aps.12.08', 'aps.12.09', 'aps.12.10', 'aps.12.11', 'aps.12.12', 'aps.12.13', 'aps.12.14', 'aps.12.15', 'aps.12.16', 'aps.12.17', 'aps.12.18', 'aps.13.00']
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'halley', smoothness=2, known_fail=known_fail)

    def test_newton(self):
        if False:
            while True:
                i = 10
        for (f, f_1, f_2) in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            x = zeros.newton(f, 3, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)
            x = zeros.newton(f, 3, x1=5, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)
            x = zeros.newton(f, 3, fprime=f_1, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)
            x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)

    def test_newton_by_name(self):
        if False:
            i = 10
            return i + 15
        'Invoke newton through root_scalar()'
        for (f, f_1, f_2) in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, fprime=f_1, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)
        for (f, f_1, f_2) in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)

    def test_secant_by_name(self):
        if False:
            print('Hello World!')
        'Invoke secant through root_scalar()'
        for (f, f_1, f_2) in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, x1=2, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)
            r = root_scalar(f, method='secant', x0=3, x1=5, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)
        for (f, f_1, f_2) in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)

    def test_halley_by_name(self):
        if False:
            return 10
        'Invoke halley through root_scalar()'
        for (f, f_1, f_2) in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='halley', x0=3, fprime=f_1, fprime2=f_2, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)

    def test_root_scalar_fail(self):
        if False:
            i = 10
            return i + 15
        message = 'fprime2 must be specified for halley'
        with pytest.raises(ValueError, match=message):
            root_scalar(f1, method='halley', fprime=f1_1, x0=3, xtol=1e-06)
        message = 'fprime must be specified for halley'
        with pytest.raises(ValueError, match=message):
            root_scalar(f1, method='halley', fprime2=f1_2, x0=3, xtol=1e-06)

    def test_array_newton(self):
        if False:
            i = 10
            return i + 15
        'test newton with array'

        def f1(x, *a):
            if False:
                for i in range(10):
                    print('nop')
            b = a[0] + x * a[3]
            return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x

        def f1_1(x, *a):
            if False:
                print('Hello World!')
            b = a[3] / a[5]
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1

        def f1_2(x, *a):
            if False:
                print('Hello World!')
            b = a[3] / a[5]
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b ** 2
        a0 = np.array([5.32725221, 5.48673747, 5.49539973, 5.36387202, 4.80237316, 1.43764452, 5.23063958, 5.46094772, 5.50512718, 5.4204629])
        a1 = (np.sin(range(10)) + 1.0) * 7.0
        args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
        x0 = [7.0] * 10
        x = zeros.newton(f1, x0, f1_1, args)
        x_expected = (6.17264965, 11.7702805, 12.2219954, 7.11017681, 1.18151293, 0.143707955, 4.31928228, 10.5419107, 12.755249, 8.91225749)
        assert_allclose(x, x_expected)
        x = zeros.newton(f1, x0, f1_1, args, fprime2=f1_2)
        assert_allclose(x, x_expected)
        x = zeros.newton(f1, x0, args=args)
        assert_allclose(x, x_expected)

    def test_array_newton_complex(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                i = 10
                return i + 15
            return x + 1 + 1j

        def fprime(x):
            if False:
                while True:
                    i = 10
            return 1.0
        t = np.full(4, 1j)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.0)
        t = np.ones(4)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.0)
        x = zeros.newton(f, t)
        assert_allclose(f(x), 0.0)

    def test_array_secant_active_zero_der(self):
        if False:
            return 10
        "test secant doesn't continue to iterate zero derivatives"
        x = zeros.newton(lambda x, *a: x * x - a[0], x0=[4.123, 5], args=[np.array([17, 25])])
        assert_allclose(x, (4.123105625617661, 5.0))

    def test_array_newton_integers(self):
        if False:
            print('Hello World!')
        x = zeros.newton(lambda y, z: z - y ** 2, [4.0] * 2, args=([15.0, 17.0],))
        assert_allclose(x, (3.872983346207417, 4.123105625617661))
        x = zeros.newton(lambda y, z: z - y ** 2, [4] * 2, args=([15, 17],))
        assert_allclose(x, (3.872983346207417, 4.123105625617661))

    def test_array_newton_zero_der_failures(self):
        if False:
            for i in range(10):
                print('nop')
        assert_warns(RuntimeWarning, zeros.newton, lambda y: y ** 2 - 2, [0.0, 0.0], lambda y: 2 * y)
        with pytest.warns(RuntimeWarning):
            results = zeros.newton(lambda y: y ** 2 - 2, [0.0, 0.0], lambda y: 2 * y, full_output=True)
            assert_allclose(results.root, 0)
            assert results.zero_der.all()
            assert not results.converged.any()

    def test_newton_combined(self):
        if False:
            print('Hello World!')

        def f1(x):
            if False:
                i = 10
                return i + 15
            return x ** 2 - 2 * x - 1

        def f1_1(x):
            if False:
                i = 10
                return i + 15
            return 2 * x - 2

        def f1_2(x):
            if False:
                while True:
                    i = 10
            return 2.0 + 0 * x

        def f1_and_p_and_pp(x):
            if False:
                return 10
            return (x ** 2 - 2 * x - 1, 2 * x - 2, 2.0)
        sol0 = root_scalar(f1, method='newton', x0=3, fprime=f1_1)
        sol = root_scalar(f1_and_p_and_pp, method='newton', x0=3, fprime=True)
        assert_allclose(sol0.root, sol.root, atol=1e-08)
        assert_equal(2 * sol.function_calls, sol0.function_calls)
        sol0 = root_scalar(f1, method='halley', x0=3, fprime=f1_1, fprime2=f1_2)
        sol = root_scalar(f1_and_p_and_pp, method='halley', x0=3, fprime2=True)
        assert_allclose(sol0.root, sol.root, atol=1e-08)
        assert_equal(3 * sol.function_calls, sol0.function_calls)

    def test_newton_full_output(self):
        if False:
            while True:
                i = 10
        x0 = 3
        expected_counts = [(6, 7), (5, 10), (3, 9)]
        for derivs in range(3):
            kwargs = {'tol': 1e-06, 'full_output': True}
            for (k, v) in [['fprime', f1_1], ['fprime2', f1_2]][:derivs]:
                kwargs[k] = v
            (x, r) = zeros.newton(f1, x0, disp=False, **kwargs)
            assert_(r.converged)
            assert_equal(x, r.root)
            assert_equal((r.iterations, r.function_calls), expected_counts[derivs])
            if derivs == 0:
                assert r.function_calls <= r.iterations + 1
            else:
                assert_equal(r.function_calls, (derivs + 1) * r.iterations)
            iters = r.iterations - 1
            (x, r) = zeros.newton(f1, x0, maxiter=iters, disp=False, **kwargs)
            assert_(not r.converged)
            assert_equal(x, r.root)
            assert_equal(r.iterations, iters)
            if derivs == 1:
                with pytest.raises(RuntimeError, match='Failed to converge after %d iterations, value is .*' % iters):
                    (x, r) = zeros.newton(f1, x0, maxiter=iters, disp=True, **kwargs)

    def test_deriv_zero_warning(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                return 10
            return x ** 2 - 2.0

        def dfunc(x):
            if False:
                while True:
                    i = 10
            return 2 * x
        assert_warns(RuntimeWarning, zeros.newton, func, 0.0, dfunc, disp=False)
        with pytest.raises(RuntimeError, match='Derivative was zero'):
            zeros.newton(func, 0.0, dfunc)

    def test_newton_does_not_modify_x0(self):
        if False:
            while True:
                i = 10
        x0 = np.array([0.1, 3])
        x0_copy = x0.copy()
        newton(np.sin, x0, np.cos)
        assert_array_equal(x0, x0_copy)

    def test_gh17570_defaults(self):
        if False:
            print('Hello World!')
        res_newton_default = root_scalar(f1, method='newton', x0=3, xtol=1e-06)
        res_secant_default = root_scalar(f1, method='secant', x0=3, x1=2, xtol=1e-06)
        res_secant = newton(f1, x0=3, x1=2, tol=1e-06, full_output=True)[1]
        assert_allclose(f1(res_newton_default.root), 0, atol=1e-06)
        assert res_newton_default.root.shape == tuple()
        assert_allclose(f1(res_secant_default.root), 0, atol=1e-06)
        assert res_secant_default.root.shape == tuple()
        assert_allclose(f1(res_secant.root), 0, atol=1e-06)
        assert res_secant.root.shape == tuple()
        assert res_secant_default.root == res_secant.root != res_newton_default.iterations
        assert res_secant_default.iterations == res_secant_default.function_calls - 1 == res_secant.iterations != res_newton_default.iterations == res_newton_default.function_calls / 2

    @pytest.mark.parametrize('kwargs', [dict(), {'method': 'newton'}])
    def test_args_gh19090(self, kwargs):
        if False:
            i = 10
            return i + 15

        def f(x, a, b):
            if False:
                while True:
                    i = 10
            assert a == 3
            assert b == 1
            return x ** a - b
        res = optimize.root_scalar(f, x0=3, args=(3, 1), **kwargs)
        assert res.converged
        assert_allclose(res.root, 1)

    @pytest.mark.parametrize('method', ['secant', 'newton'])
    def test_int_x0_gh19280(self, method):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                i = 10
                return i + 15
            return x ** (-2) - 2
        res = optimize.root_scalar(f, x0=1, method=method)
        assert res.converged
        assert_allclose(abs(res.root), 2 ** (-0.5))
        assert res.root.dtype == np.dtype(np.float64)

def test_gh_5555():
    if False:
        i = 10
        return i + 15
    root = 0.1

    def f(x):
        if False:
            print('Hello World!')
        return x - root
    methods = [zeros.bisect, zeros.ridder]
    xtol = rtol = TOL
    for method in methods:
        res = method(f, -100000000.0, 10000000.0, xtol=xtol, rtol=rtol)
        assert_allclose(root, res, atol=xtol, rtol=rtol, err_msg='method %s' % method.__name__)

def test_gh_5557():
    if False:
        for i in range(10):
            print('nop')

    def f(x):
        if False:
            while True:
                i = 10
        if x < 0.5:
            return -0.1
        else:
            return x - 0.6
    atol = 0.51
    rtol = 4 * _FLOAT_EPS
    methods = [zeros.brentq, zeros.brenth]
    for method in methods:
        res = method(f, 0, 1, xtol=atol, rtol=rtol)
        assert_allclose(0.6, res, atol=atol, rtol=rtol)

def test_brent_underflow_in_root_bracketing():
    if False:
        for i in range(10):
            print('nop')
    underflow_scenario = (-450.0, -350.0, -400.0)
    overflow_scenario = (350.0, 450.0, 400.0)
    for (a, b, root) in [underflow_scenario, overflow_scenario]:
        c = np.exp(root)
        for method in [zeros.brenth, zeros.brentq]:
            res = method(lambda x: np.exp(x) - c, a, b)
            assert_allclose(root, res)

class TestRootResults:
    r = zeros.RootResults(root=1.0, iterations=44, function_calls=46, flag=0, method='newton')

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        expected_repr = '      converged: True\n           flag: converged\n function_calls: 46\n     iterations: 44\n           root: 1.0\n         method: newton'
        assert_equal(repr(self.r), expected_repr)

    def test_type(self):
        if False:
            i = 10
            return i + 15
        assert isinstance(self.r, OptimizeResult)

def test_complex_halley():
    if False:
        print('Hello World!')
    "Test Halley's works with complex roots"

    def f(x, *a):
        if False:
            while True:
                i = 10
        return a[0] * x ** 2 + a[1] * x + a[2]

    def f_1(x, *a):
        if False:
            while True:
                i = 10
        return 2 * a[0] * x + a[1]

    def f_2(x, *a):
        if False:
            return 10
        retval = 2 * a[0]
        try:
            size = len(x)
        except TypeError:
            return retval
        else:
            return [retval] * size
    z = complex(1.0, 2.0)
    coeffs = (2.0, 3.0, 4.0)
    y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-06)
    assert_allclose(f(y, *coeffs), 0, atol=1e-06)
    z = [z] * 10
    coeffs = (2.0, 3.0, 4.0)
    y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-06)
    assert_allclose(f(y, *coeffs), 0, atol=1e-06)

def test_zero_der_nz_dp():
    if False:
        print('Hello World!')
    'Test secant method with a non-zero dp, but an infinite newton step'
    dx = np.finfo(float).eps ** 0.33
    p0 = (200.0 - dx) / (2.0 + dx)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'RMS of')
        x = zeros.newton(lambda y: (y - 100.0) ** 2, x0=[p0] * 10)
    assert_allclose(x, [100] * 10)
    p0 = (2.0 - 0.0001) / (2.0 + 0.0001)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Tolerance of')
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=False)
    assert_allclose(x, 1)
    with pytest.raises(RuntimeError, match='Tolerance of'):
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=True)
    p0 = (-2.0 + 0.0001) / (2.0 + 0.0001)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Tolerance of')
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=False)
    assert_allclose(x, -1)
    with pytest.raises(RuntimeError, match='Tolerance of'):
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=True)

def test_array_newton_failures():
    if False:
        print('Hello World!')
    'Test that array newton fails as expected'
    diameter = 0.1
    roughness = 0.00015
    rho = 988.1
    mu = 0.0005479
    u = 2.488
    reynolds_number = rho * u * diameter / mu

    def colebrook_eqn(darcy_friction, re, dia):
        if False:
            while True:
                i = 10
        return 1 / np.sqrt(darcy_friction) + 2 * np.log10(roughness / 3.7 / dia + 2.51 / re / np.sqrt(darcy_friction))
    with pytest.warns(RuntimeWarning):
        result = zeros.newton(colebrook_eqn, x0=[0.01, 0.2, 0.02223, 0.3], maxiter=2, args=[reynolds_number, diameter], full_output=True)
        assert not result.converged.all()
    with pytest.raises(RuntimeError):
        result = zeros.newton(colebrook_eqn, x0=[0.01] * 2, maxiter=2, args=[reynolds_number, diameter], full_output=True)

def test_gh8904_zeroder_at_root_fails():
    if False:
        i = 10
        return i + 15
    "Test that Newton or Halley don't warn if zero derivative at root"

    def f_zeroder_root(x):
        if False:
            while True:
                i = 10
        return x ** 3 - x ** 2
    r = zeros.newton(f_zeroder_root, x0=0)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0] * 10)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

    def fder(x):
        if False:
            for i in range(10):
                print('nop')
        return 3 * x ** 2 - 2 * x

    def fder2(x):
        if False:
            return 10
        return 6 * x - 2
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder, fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0] * 10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0] * 10, fprime=fder, fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=0.5, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0.5] * 10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

def test_gh_8881():
    if False:
        i = 10
        return i + 15
    "Test that Halley's method realizes that the 2nd order adjustment\n    is too big and drops off to the 1st order adjustment."
    n = 9

    def f(x):
        if False:
            print('Hello World!')
        return power(x, 1.0 / n) - power(n, 1.0 / n)

    def fp(x):
        if False:
            while True:
                i = 10
        return power(x, (1.0 - n) / n) / n

    def fpp(x):
        if False:
            while True:
                i = 10
        return power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n
    x0 = 0.1
    (rt, r) = newton(f, x0, fprime=fp, full_output=True)
    assert r.converged
    (rt, r) = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert r.converged

def test_gh_9608_preserve_array_shape():
    if False:
        print('Hello World!')
    '\n    Test that shape is preserved for array inputs even if fprime or fprime2 is\n    scalar\n    '

    def f(x):
        if False:
            while True:
                i = 10
        return x ** 2

    def fp(x):
        if False:
            while True:
                i = 10
        return 2 * x

    def fpp(x):
        if False:
            i = 10
            return i + 15
        return 2
    x0 = np.array([-2], dtype=np.float32)
    (rt, r) = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert r.converged
    x0_array = np.array([-2, -3], dtype=np.float32)
    with pytest.raises(IndexError):
        result = zeros.newton(f, x0_array, fprime=fp, fprime2=fpp, full_output=True)

    def fpp_array(x):
        if False:
            for i in range(10):
                print('nop')
        return np.full(np.shape(x), 2, dtype=np.float32)
    result = zeros.newton(f, x0_array, fprime=fp, fprime2=fpp_array, full_output=True)
    assert result.converged.all()

@pytest.mark.parametrize('maximum_iterations,flag_expected', [(10, zeros.CONVERR), (100, zeros.CONVERGED)])
def test_gh9254_flag_if_maxiter_exceeded(maximum_iterations, flag_expected):
    if False:
        while True:
            i = 10
    '\n    Test that if the maximum iterations is exceeded that the flag is not\n    converged.\n    '
    result = zeros.brentq(lambda x: ((1.2 * x - 2.3) * x + 3.4) * x - 4.5, -30, 30, (), 1e-06, 1e-06, maximum_iterations, full_output=True, disp=False)
    assert result[1].flag == flag_expected
    if flag_expected == zeros.CONVERR:
        assert result[1].iterations == maximum_iterations
    elif flag_expected == zeros.CONVERGED:
        assert result[1].iterations < maximum_iterations

def test_gh9551_raise_error_if_disp_true():
    if False:
        return 10
    'Test that if disp is true then zero derivative raises RuntimeError'

    def f(x):
        if False:
            print('Hello World!')
        return x * x + 1

    def f_p(x):
        if False:
            for i in range(10):
                print('nop')
        return 2 * x
    assert_warns(RuntimeWarning, zeros.newton, f, 1.0, f_p, disp=False)
    with pytest.raises(RuntimeError, match='^Derivative was zero\\. Failed to converge after \\d+ iterations, value is [+-]?\\d*\\.\\d+\\.$'):
        zeros.newton(f, 1.0, f_p)
    root = zeros.newton(f, complex(10.0, 10.0), f_p)
    assert_allclose(root, complex(0.0, 1.0))

@pytest.mark.parametrize('solver_name', ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
def test_gh3089_8394(solver_name):
    if False:
        while True:
            i = 10

    def f(x):
        if False:
            while True:
                i = 10
        return np.nan
    solver = getattr(zeros, solver_name)
    with pytest.raises(ValueError, match='The function value at x...'):
        solver(f, 0, 1)

@pytest.mark.parametrize('method', ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
def test_gh18171(method):
    if False:
        i = 10
        return i + 15

    def f(x):
        if False:
            for i in range(10):
                print('nop')
        f._count += 1
        return np.nan
    f._count = 0
    res = root_scalar(f, bracket=(0, 1), method=method)
    assert res.converged is False
    assert res.flag.startswith('The function value at x')
    assert res.function_calls == f._count
    assert str(res.root) in res.flag

@pytest.mark.parametrize('solver_name', ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
@pytest.mark.parametrize('rs_interface', [True, False])
def test_function_calls(solver_name, rs_interface):
    if False:
        return 10
    solver = (lambda f, a, b, **kwargs: root_scalar(f, bracket=(a, b))) if rs_interface else getattr(zeros, solver_name)

    def f(x):
        if False:
            print('Hello World!')
        f.calls += 1
        return x ** 2 - 1
    f.calls = 0
    res = solver(f, 0, 10, full_output=True)
    if rs_interface:
        assert res.function_calls == f.calls
    else:
        assert res[1].function_calls == f.calls

def test_gh_14486_converged_false():
    if False:
        for i in range(10):
            print('nop')
    'Test that zero slope with secant method results in a converged=False'

    def lhs(x):
        if False:
            i = 10
            return i + 15
        return x * np.exp(-x * x) - 0.07
    with pytest.warns(RuntimeWarning, match='Tolerance of'):
        res = root_scalar(lhs, method='secant', x0=-0.15, x1=1.0)
    assert not res.converged
    assert res.flag == 'convergence error'
    with pytest.warns(RuntimeWarning, match='Tolerance of'):
        res = newton(lhs, x0=-0.15, x1=1.0, disp=False, full_output=True)[1]
    assert not res.converged
    assert res.flag == 'convergence error'

@pytest.mark.parametrize('solver_name', ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
@pytest.mark.parametrize('rs_interface', [True, False])
def test_gh5584(solver_name, rs_interface):
    if False:
        return 10
    solver = (lambda f, a, b, **kwargs: root_scalar(f, bracket=(a, b))) if rs_interface else getattr(zeros, solver_name)

    def f(x):
        if False:
            return 10
        return 1e-200 * x
    with pytest.raises(ValueError, match='...must have different signs'):
        solver(f, -0.5, -0.4, full_output=True)
    res = solver(f, -0.5, 0.4, full_output=True)
    res = res if rs_interface else res[1]
    assert res.converged
    assert_allclose(res.root, 0, atol=1e-08)
    res = solver(f, -0.5, float('-0.0'), full_output=True)
    res = res if rs_interface else res[1]
    assert res.converged
    assert_allclose(res.root, 0, atol=1e-08)

def test_gh13407():
    if False:
        for i in range(10):
            print('nop')

    def f(x):
        if False:
            return 10
        return x ** 3 - 2 * x - 5
    xtol = 1e-300
    eps = np.finfo(float).eps
    x1 = zeros.toms748(f, 1e-10, 10000000000.0, xtol=xtol, rtol=1 * eps)
    f1 = f(x1)
    x4 = zeros.toms748(f, 1e-10, 10000000000.0, xtol=xtol, rtol=4 * eps)
    f4 = f(x4)
    assert f1 < f4
    message = f'rtol too small \\({eps / 2:g} < {eps:g}\\)'
    with pytest.raises(ValueError, match=message):
        zeros.toms748(f, 1e-10, 10000000000.0, xtol=xtol, rtol=eps / 2)

def test_newton_complex_gh10103():
    if False:
        return 10

    def f(z):
        if False:
            return 10
        return z - 1
    res = newton(f, 1 + 1j)
    assert_allclose(res, 1, atol=1e-12)
    res = root_scalar(f, x0=1 + 1j, x1=2 + 1.5j, method='secant')
    assert_allclose(res.root, 1, atol=1e-12)

@pytest.mark.parametrize('method', all_methods)
def test_maxiter_int_check_gh10236(method):
    if False:
        for i in range(10):
            print('nop')
    message = "'float' object cannot be interpreted as an integer"
    with pytest.raises(TypeError, match=message):
        method(f1, 0.0, 1.0, maxiter=72.45)

class TestDifferentiate:

    def f(self, x):
        if False:
            while True:
                i = 10
        return stats.norm().cdf(x)

    @pytest.mark.parametrize('x', [0.6, np.linspace(-0.05, 1.05, 10)])
    def test_basic(self, x):
        if False:
            return 10
        res = zeros._differentiate(self.f, x)
        ref = stats.norm().pdf(x)
        np.testing.assert_allclose(res.df, ref)
        assert_array_less(abs(res.df - ref), res.error)
        assert res.x.shape == ref.shape

    @pytest.mark.parametrize('case', stats._distr_params.distcont)
    def test_accuracy(self, case):
        if False:
            while True:
                i = 10
        (distname, params) = case
        dist = getattr(stats, distname)(*params)
        x = dist.median() + 0.1
        res = zeros._differentiate(dist.cdf, x)
        ref = dist.pdf(x)
        assert_allclose(res.df, ref, atol=1e-10)

    @pytest.mark.parametrize('order', [1, 6])
    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, order, shape):
        if False:
            while True:
                i = 10
        x = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        n = np.size(x)

        @np.vectorize
        def _differentiate_single(x):
            if False:
                return 10
            return zeros._differentiate(self.f, x, order=order)

        def f(x, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            f.nit += 1
            f.feval += 1 if x.size == n or x.ndim <= 1 else x.shape[-1]
            return self.f(x, *args, **kwargs)
        f.nit = -1
        f.feval = 0
        res = zeros._differentiate(f, x, order=order)
        refs = _differentiate_single(x).ravel()
        ref_x = [ref.x for ref in refs]
        assert_allclose(res.x.ravel(), ref_x)
        assert_equal(res.x.shape, shape)
        ref_df = [ref.df for ref in refs]
        assert_allclose(res.df.ravel(), ref_df)
        assert_equal(res.df.shape, shape)
        ref_error = [ref.error for ref in refs]
        assert_allclose(res.error.ravel(), ref_error, atol=5e-15)
        assert_equal(res.error.shape, shape)
        ref_success = [ref.success for ref in refs]
        assert_equal(res.success.ravel(), ref_success)
        assert_equal(res.success.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        ref_flag = [ref.status for ref in refs]
        assert_equal(res.status.ravel(), ref_flag)
        assert_equal(res.status.shape, shape)
        assert np.issubdtype(res.status.dtype, np.integer)
        ref_nfev = [ref.nfev for ref in refs]
        assert_equal(res.nfev.ravel(), ref_nfev)
        assert_equal(np.max(res.nfev), f.feval)
        assert_equal(res.nfev.shape, res.x.shape)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        ref_nit = [ref.nit for ref in refs]
        assert_equal(res.nit.ravel(), ref_nit)
        assert_equal(np.max(res.nit), f.nit)
        assert_equal(res.nit.shape, res.x.shape)
        assert np.issubdtype(res.nit.dtype, np.integer)

    def test_flags(self):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(5651219684984213)

        def f(xs, js):
            if False:
                while True:
                    i = 10
            f.nit += 1
            funcs = [lambda x: x - 2.5, lambda x: np.exp(x) * rng.random(), lambda x: np.exp(x), lambda x: np.full_like(x, np.nan)[()]]
            res = [funcs[j](x) for (x, j) in zip(xs, js.ravel())]
            return res
        f.nit = 0
        args = (np.arange(4, dtype=np.int64),)
        res = zeros._differentiate(f, [1] * 4, rtol=1e-14, order=2, args=args)
        ref_flags = np.array([zeros._ECONVERGED, zeros._EERRORINCREASE, zeros._ECONVERR, zeros._EVALUEERR])
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        if False:
            for i in range(10):
                print('nop')
        dist = stats.norm()
        x = 1
        f = dist.cdf
        ref = dist.pdf(x)
        kwargs0 = dict(atol=0, rtol=0, order=4)
        kwargs = kwargs0.copy()
        kwargs['atol'] = 0.001
        res1 = zeros._differentiate(f, x, **kwargs)
        assert_array_less(abs(res1.df - ref), 0.001)
        kwargs['atol'] = 1e-06
        res2 = zeros._differentiate(f, x, **kwargs)
        assert_array_less(abs(res2.df - ref), 1e-06)
        assert_array_less(abs(res2.df - ref), abs(res1.df - ref))
        kwargs = kwargs0.copy()
        kwargs['rtol'] = 0.001
        res1 = zeros._differentiate(f, x, **kwargs)
        assert_array_less(abs(res1.df - ref), 0.001 * np.abs(ref))
        kwargs['rtol'] = 1e-06
        res2 = zeros._differentiate(f, x, **kwargs)
        assert_array_less(abs(res2.df - ref), 1e-06 * np.abs(ref))
        assert_array_less(abs(res2.df - ref), abs(res1.df - ref))

    def test_step_parameters(self):
        if False:
            return 10
        dist = stats.norm()
        x = 1
        f = dist.cdf
        ref = dist.pdf(x)
        res1 = zeros._differentiate(f, x, initial_step=0.5, maxiter=1)
        res2 = zeros._differentiate(f, x, initial_step=0.05, maxiter=1)
        assert abs(res2.df - ref) < abs(res1.df - ref)
        res1 = zeros._differentiate(f, x, step_factor=2, maxiter=1)
        res2 = zeros._differentiate(f, x, step_factor=20, maxiter=1)
        assert abs(res2.df - ref) < abs(res1.df - ref)
        kwargs = dict(order=4, maxiter=1, step_direction=0)
        res = zeros._differentiate(f, x, initial_step=0.5, step_factor=0.5, **kwargs)
        ref = zeros._differentiate(f, x, initial_step=1, step_factor=2, **kwargs)
        assert_allclose(res.df, ref.df, rtol=5e-15)
        kwargs = dict(order=2, maxiter=1, step_direction=1)
        res = zeros._differentiate(f, x, initial_step=1, step_factor=2, **kwargs)
        ref = zeros._differentiate(f, x, initial_step=1 / np.sqrt(2), step_factor=0.5, **kwargs)
        assert_allclose(res.df, ref.df, rtol=5e-15)
        kwargs['step_direction'] = -1
        res = zeros._differentiate(f, x, initial_step=1, step_factor=2, **kwargs)
        ref = zeros._differentiate(f, x, initial_step=1 / np.sqrt(2), step_factor=0.5, **kwargs)
        assert_allclose(res.df, ref.df, rtol=5e-15)

    def test_step_direction(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                print('Hello World!')
            y = np.exp(x)
            y[(x < 0) + (x > 2)] = np.nan
            return y
        x = np.linspace(0, 2, 10)
        step_direction = np.zeros_like(x)
        (step_direction[x < 0.6], step_direction[x > 1.4]) = (1, -1)
        res = zeros._differentiate(f, x, step_direction=step_direction)
        assert_allclose(res.df, np.exp(x))
        assert np.all(res.success)

    def test_vectorized_step_direction_args(self):
        if False:
            print('Hello World!')

        def f(x, p):
            if False:
                return 10
            return x ** p

        def df(x, p):
            if False:
                return 10
            return p * x ** (p - 1)
        x = np.array([1, 2, 3, 4]).reshape(-1, 1, 1)
        hdir = np.array([-1, 0, 1]).reshape(1, -1, 1)
        p = np.array([2, 3]).reshape(1, 1, -1)
        res = zeros._differentiate(f, x, step_direction=hdir, args=(p,))
        ref = np.broadcast_to(df(x, p), res.df.shape)
        assert_allclose(res.df, ref)

    def test_maxiter_callback(self):
        if False:
            for i in range(10):
                print('nop')
        x = 0.612814
        dist = stats.norm()
        maxiter = 3

        def f(x):
            if False:
                while True:
                    i = 10
            res = dist.cdf(x)
            return res
        default_order = 8
        res = zeros._differentiate(f, x, maxiter=maxiter, rtol=1e-15)
        assert not np.any(res.success)
        assert np.all(res.nfev == default_order + 1 + (maxiter - 1) * 2)
        assert np.all(res.nit == maxiter)

        def callback(res):
            if False:
                i = 10
                return i + 15
            callback.iter += 1
            callback.res = res
            assert hasattr(res, 'x')
            assert res.df not in callback.dfs
            callback.dfs.add(res.df)
            assert res.status == zeros._EINPROGRESS
            if callback.iter == maxiter:
                raise StopIteration
        callback.iter = -1
        callback.res = None
        callback.dfs = set()
        res2 = zeros._differentiate(f, x, callback=callback, rtol=1e-15)
        for key in res.keys():
            if key == 'status':
                assert res[key] == zeros._ECONVERR
                assert callback.res[key] == zeros._EINPROGRESS
                assert res2[key] == zeros._ECALLBACK
            else:
                assert res2[key] == callback.res[key] == res[key]

    @pytest.mark.parametrize('hdir', (-1, 0, 1))
    @pytest.mark.parametrize('x', (0.65, [0.65, 0.7]))
    @pytest.mark.parametrize('dtype', (np.float16, np.float32, np.float64))
    def test_dtype(self, hdir, x, dtype):
        if False:
            i = 10
            return i + 15
        x = np.asarray(x, dtype=dtype)[()]

        def f(x):
            if False:
                i = 10
                return i + 15
            assert x.dtype == dtype
            return np.exp(x)

        def callback(res):
            if False:
                while True:
                    i = 10
            assert res.x.dtype == dtype
            assert res.df.dtype == dtype
            assert res.error.dtype == dtype
        res = zeros._differentiate(f, x, order=4, step_direction=hdir, callback=callback)
        assert res.x.dtype == dtype
        assert res.df.dtype == dtype
        assert res.error.dtype == dtype
        eps = np.finfo(dtype).eps
        assert_allclose(res.df, np.exp(res.x), rtol=np.sqrt(eps))

    def test_input_validation(self):
        if False:
            print('Hello World!')
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(None, 1)
        message = 'Abscissae and function output must be real numbers.'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, -4 + 1j)
        message = 'The shape of the array returned by `func`'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: [1, 2, 3], [-2, -3])
        message = 'Tolerances and step parameters must be non-negative...'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, atol=-1)
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, rtol='ekki')
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, initial_step=None)
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, step_factor=object())
        message = '`maxiter` must be a positive integer.'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, maxiter=0)
        message = '`order` must be a positive integer'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, order=1.5)
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, order=0)
        message = '`callback` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._differentiate(lambda x: x, 1, callback='shrubbery')

    def test_special_cases(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99 - 1
        res = zeros._differentiate(f, 7, rtol=1e-10)
        assert res.success
        assert_allclose(res.df, 99 * 7.0 ** 98)
        for n in range(6):
            x = 1.5

            def f(x):
                if False:
                    return 10
                return 2 * x ** n
            ref = 2 * n * x ** (n - 1)
            res = zeros._differentiate(f, x, maxiter=1, order=max(1, n))
            assert_allclose(res.df, ref, rtol=1e-15)
            assert_equal(res.error, np.nan)
            res = zeros._differentiate(f, x, order=max(1, n))
            assert res.success
            assert res.nit == 2
            assert_allclose(res.df, ref, rtol=1e-15)

        def f(x, c):
            if False:
                for i in range(10):
                    print('nop')
            return c * x - 1
        res = zeros._differentiate(f, 2, args=3)
        assert_allclose(res.df, 3)

    @pytest.mark.xfail
    @pytest.mark.parametrize('case', ((lambda x: (x - 1) ** 3, 1), (lambda x: np.where(x > 1, (x - 1) ** 5, (x - 1) ** 3), 1)))
    def test_saddle_gh18811(self, case):
        if False:
            while True:
                i = 10
        atol = 1e-16
        res = zeros._differentiate(*case, step_direction=[-1, 0, 1], atol=atol)
        assert np.all(res.success)
        assert_allclose(res.df, 0, atol=atol)

class TestBracketRoot:

    @pytest.mark.parametrize('seed', (615655101, 3141866013, 238075752))
    @pytest.mark.parametrize('use_min', (False, True))
    @pytest.mark.parametrize('other_side', (False, True))
    @pytest.mark.parametrize('fix_one_side', (False, True))
    def test_nfev_expected(self, seed, use_min, other_side, fix_one_side):
        if False:
            i = 10
            return i + 15
        rng = np.random.default_rng(seed)
        (a, d, factor) = rng.random(size=3) * [100000.0, 10, 5]
        factor = 1 + factor
        b = a + d

        def f(x):
            if False:
                i = 10
                return i + 15
            f.count += 1
            return x
        if use_min:
            min = -rng.random()
            n = np.ceil(np.log(-(a - min) / min) / np.log(factor))
            (l, u) = (min + (a - min) * factor ** (-n), min + (a - min) * factor ** (-(n - 1)))
            kwargs = dict(a=a, b=b, factor=factor, min=min)
        else:
            n = np.ceil(np.log(b / d) / np.log(factor))
            (l, u) = (b - d * factor ** n, b - d * factor ** (n - 1))
            kwargs = dict(a=a, b=b, factor=factor)
        if other_side:
            (kwargs['a'], kwargs['b']) = (-kwargs['b'], -kwargs['a'])
            (l, u) = (-u, -l)
            if 'min' in kwargs:
                kwargs['max'] = -kwargs.pop('min')
        if fix_one_side:
            if other_side:
                kwargs['min'] = -b
            else:
                kwargs['max'] = b
        f.count = 0
        res = zeros._bracket_root(f, **kwargs)
        if not fix_one_side:
            assert res.nfev == 2 * (res.nit + 1) == 2 * (f.count - 1) == 2 * (n + 1)
        else:
            assert res.nfev == res.nit + 1 + 1 == f.count - 1 + 1 == n + 1 + 1
        bracket = np.asarray([res.xl, res.xr])
        assert_allclose(bracket, (l, u))
        f_bracket = np.asarray([res.fl, res.fr])
        assert_allclose(f_bracket, f(bracket))
        assert res.xr > res.xl
        signs = np.sign(f_bracket)
        assert signs[0] == -signs[1]
        assert res.status == 0
        assert res.success

    def f(self, q, p):
        if False:
            print('Hello World!')
        return stats.norm.cdf(q) - p

    @pytest.mark.parametrize('p', [0.6, np.linspace(0.05, 0.95, 10)])
    @pytest.mark.parametrize('min', [-5, None])
    @pytest.mark.parametrize('max', [5, None])
    @pytest.mark.parametrize('factor', [1.2, 2])
    def test_basic(self, p, min, max, factor):
        if False:
            return 10
        res = zeros._bracket_root(self.f, -0.01, 0.01, min=min, max=max, factor=factor, args=(p,))
        assert_equal(-np.sign(res.fl), np.sign(res.fr))

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        if False:
            i = 10
            return i + 15
        p = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        args = (p,)
        maxiter = 10

        @np.vectorize
        def bracket_root_single(a, b, min, max, factor, p):
            if False:
                return 10
            return zeros._bracket_root(self.f, a, b, min=min, max=max, factor=factor, args=(p,), maxiter=maxiter)

        def f(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            f.f_evals += 1
            return self.f(*args, **kwargs)
        f.f_evals = 0
        rng = np.random.default_rng(2348234)
        a = -rng.random(size=shape)
        b = rng.random(size=shape)
        (min, max) = (1000.0 * a, 1000.0 * b)
        if shape:
            i = rng.random(size=shape) > 0.5
            (min[i], max[i]) = (-np.inf, np.inf)
        factor = rng.random(size=shape) + 1.5
        res = zeros._bracket_root(f, a, b, min=min, max=max, factor=factor, args=args, maxiter=maxiter)
        refs = bracket_root_single(a, b, min, max, factor, p).ravel()
        attrs = ['xl', 'xr', 'fl', 'fr', 'success', 'nfev', 'nit']
        for attr in attrs:
            ref_attr = [getattr(ref, attr) for ref in refs]
            res_attr = getattr(res, attr)
            assert_allclose(res_attr.ravel(), ref_attr)
            assert_equal(res_attr.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        if shape:
            assert np.all(res.success[1:-1])
        assert np.issubdtype(res.status.dtype, np.integer)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        assert np.issubdtype(res.nit.dtype, np.integer)
        assert_equal(np.max(res.nit), f.f_evals - 2)
        assert_array_less(res.xl, res.xr)
        assert_allclose(res.fl, self.f(res.xl, *args))
        assert_allclose(res.fr, self.f(res.xr, *args))

    def test_flags(self):
        if False:
            while True:
                i = 10

        def f(xs, js):
            if False:
                return 10
            funcs = [lambda x: x - 1.5, lambda x: x - 1000, lambda x: x - 1000, lambda x: np.nan]
            return [funcs[j](x) for (x, j) in zip(xs, js)]
        args = (np.arange(4, dtype=np.int64),)
        res = zeros._bracket_root(f, a=[-1, -1, -1, -1], b=[1, 1, 1, 1], min=[-np.inf, -1, -np.inf, -np.inf], max=[np.inf, 1, np.inf, np.inf], args=args, maxiter=3)
        ref_flags = np.array([zeros._ECONVERGED, zeros._ELIMITS, zeros._ECONVERR, zeros._EVALUEERR])
        assert_equal(res.status, ref_flags)

    @pytest.mark.parametrize('root', (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize('min', [-5, None])
    @pytest.mark.parametrize('max', [5, None])
    @pytest.mark.parametrize('dtype', (np.float16, np.float32, np.float64))
    def test_dtype(self, root, min, max, dtype):
        if False:
            i = 10
            return i + 15
        min = min if min is None else dtype(min)
        max = max if max is None else dtype(max)
        root = dtype(root)

        def f(x, root):
            if False:
                return 10
            return ((x - root) ** 3).astype(dtype)
        bracket = np.asarray([-0.01, 0.01], dtype=dtype)
        res = zeros._bracket_root(f, *bracket, min=min, max=max, args=(root,))
        assert np.all(res.success)
        assert res.xl.dtype == res.xr.dtype == dtype
        assert res.fl.dtype == res.fr.dtype == dtype

    def test_input_validation(self):
        if False:
            return 10
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(None, -4, 4)
        message = '...must be numeric and real.'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4 + 1j, 4)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 'hello')
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, min=zeros)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, max=object())
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, factor=sum)
        message = 'All elements of `factor` must be greater than 1.'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, factor=0.5)
        message = '`min <= a < b <= max` must be True'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, 4, -4)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, max=np.nan)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, min=10)
        message = 'shape mismatch: objects cannot be broadcast'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, [-2, -3], [3, 4, 5])
        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=-1)

    def test_special_cases(self):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99 - 1
        res = zeros._bracket_root(f, -7, 5)
        assert res.success

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x - 10
        bracket = (-3, 5)
        res = zeros._bracket_root(f, *bracket, maxiter=0)
        assert res.xl, res.xr == bracket
        assert res.nit == 0
        assert res.nfev == 2
        assert res.status == -2

        def f(x, c):
            if False:
                return 10
            return c * x - 1
        res = zeros._bracket_root(f, -1, 1, args=3)
        assert res.success
        assert_allclose(res.fl, f(res.xl, 3))

        def f(x):
            if False:
                return 10
            f.count += 1
            return x
        f.count = 0
        zeros._bracket_root(f, -10, 20)
        assert_equal(f.count, 2)
        f.count = 0
        res = zeros._bracket_root(f, 5, 10, factor=2)
        bracket = (res.xl, res.xr)
        assert_equal(res.nfev, 4)
        assert_allclose(bracket, (0, 5), atol=1e-15)
        with np.errstate(over='ignore'):
            res = zeros._bracket_root(f, 5, 10, min=0)
        bracket = (res.xl, res.xr)
        assert_allclose(bracket[0], 0, atol=1e-15)
        with np.errstate(over='ignore'):
            res = zeros._bracket_root(f, -10, -5, max=0)
        bracket = (res.xl, res.xr)
        assert_allclose(bracket[1], 0, atol=1e-15)
        with np.errstate(over='ignore'):
            res = zeros._bracket_root(f, 5, 10, min=1)
        assert not res.success