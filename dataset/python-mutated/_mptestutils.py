import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
try:
    import mpmath
except ImportError:
    pass

class Arg:
    """Generate a set of numbers on the real axis, concentrating on
    'interesting' regions and covering all orders of magnitude.

    """

    def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
        if False:
            print('Hello World!')
        if a > b:
            raise ValueError('a should be less than or equal to b')
        if a == -np.inf:
            a = -0.5 * np.finfo(float).max
        if b == np.inf:
            b = 0.5 * np.finfo(float).max
        (self.a, self.b) = (a, b)
        (self.inclusive_a, self.inclusive_b) = (inclusive_a, inclusive_b)

    def _positive_values(self, a, b, n):
        if False:
            while True:
                i = 10
        if a < 0:
            raise ValueError('a should be positive')
        if n % 2 == 0:
            nlogpts = n // 2
            nlinpts = nlogpts
        else:
            nlogpts = n // 2
            nlinpts = nlogpts + 1
        if a >= 10:
            pts = np.logspace(np.log10(a), np.log10(b), n)
        elif a > 0 and b < 10:
            pts = np.linspace(a, b, n)
        elif a > 0:
            linpts = np.linspace(a, 10, nlinpts, endpoint=False)
            logpts = np.logspace(1, np.log10(b), nlogpts)
            pts = np.hstack((linpts, logpts))
        elif a == 0 and b <= 10:
            linpts = np.linspace(0, b, nlinpts)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts = np.logspace(-30, right, nlogpts, endpoint=False)
            pts = np.hstack((logpts, linpts))
        else:
            if nlogpts % 2 == 0:
                nlogpts1 = nlogpts // 2
                nlogpts2 = nlogpts1
            else:
                nlogpts1 = nlogpts // 2
                nlogpts2 = nlogpts1 + 1
            linpts = np.linspace(0, 10, nlinpts, endpoint=False)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts1 = np.logspace(-30, right, nlogpts1, endpoint=False)
            logpts2 = np.logspace(1, np.log10(b), nlogpts2)
            pts = np.hstack((logpts1, linpts, logpts2))
        return np.sort(pts)

    def values(self, n):
        if False:
            print('Hello World!')
        'Return an array containing n numbers.'
        (a, b) = (self.a, self.b)
        if a == b:
            return np.zeros(n)
        if not self.inclusive_a:
            n += 1
        if not self.inclusive_b:
            n += 1
        if n % 2 == 0:
            n1 = n // 2
            n2 = n1
        else:
            n1 = n // 2
            n2 = n1 + 1
        if a >= 0:
            pospts = self._positive_values(a, b, n)
            negpts = []
        elif b <= 0:
            pospts = []
            negpts = -self._positive_values(-b, -a, n)
        else:
            pospts = self._positive_values(0, b, n1)
            negpts = -self._positive_values(0, -a, n2 + 1)
            negpts = negpts[1:]
        pts = np.hstack((negpts[::-1], pospts))
        if not self.inclusive_a:
            pts = pts[1:]
        if not self.inclusive_b:
            pts = pts[:-1]
        return pts

class FixedArg:

    def __init__(self, values):
        if False:
            return 10
        self._values = np.asarray(values)

    def values(self, n):
        if False:
            i = 10
            return i + 15
        return self._values

class ComplexArg:

    def __init__(self, a=complex(-np.inf, -np.inf), b=complex(np.inf, np.inf)):
        if False:
            for i in range(10):
                print('nop')
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)

    def values(self, n):
        if False:
            print('Hello World!')
        m = int(np.floor(np.sqrt(n)))
        x = self.real.values(m)
        y = self.imag.values(m + 1)
        return (x[:, None] + 1j * y[None, :]).ravel()

class IntArg:

    def __init__(self, a=-1000, b=1000):
        if False:
            print('Hello World!')
        self.a = a
        self.b = b

    def values(self, n):
        if False:
            return 10
        v1 = Arg(self.a, self.b).values(max(1 + n // 2, n - 5)).astype(int)
        v2 = np.arange(-5, 5)
        v = np.unique(np.r_[v1, v2])
        v = v[(v >= self.a) & (v < self.b)]
        return v

def get_args(argspec, n):
    if False:
        while True:
            i = 10
    if isinstance(argspec, np.ndarray):
        args = argspec.copy()
    else:
        nargs = len(argspec)
        ms = np.asarray([1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec])
        ms = (n ** (ms / sum(ms))).astype(int) + 1
        args = [spec.values(m) for (spec, m) in zip(argspec, ms)]
        args = np.array(np.broadcast_arrays(*np.ix_(*args))).reshape(nargs, -1).T
    return args

class MpmathData:

    def __init__(self, scipy_func, mpmath_func, arg_spec, name=None, dps=None, prec=None, n=None, rtol=1e-07, atol=1e-300, ignore_inf_sign=False, distinguish_nan_and_inf=True, nan_ok=True, param_filter=None):
        if False:
            print('Hello World!')
        if n is None:
            try:
                is_xslow = int(os.environ.get('SCIPY_XSLOW', '0'))
            except ValueError:
                is_xslow = False
            n = 5000 if is_xslow else 500
        self.scipy_func = scipy_func
        self.mpmath_func = mpmath_func
        self.arg_spec = arg_spec
        self.dps = dps
        self.prec = prec
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.ignore_inf_sign = ignore_inf_sign
        self.nan_ok = nan_ok
        if isinstance(self.arg_spec, np.ndarray):
            self.is_complex = np.issubdtype(self.arg_spec.dtype, np.complexfloating)
        else:
            self.is_complex = any([isinstance(arg, ComplexArg) for arg in self.arg_spec])
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf
        if not name or name == '<lambda>':
            name = getattr(scipy_func, '__name__', None)
        if not name or name == '<lambda>':
            name = getattr(mpmath_func, '__name__', None)
        self.name = name
        self.param_filter = param_filter

    def check(self):
        if False:
            while True:
                i = 10
        np.random.seed(1234)
        argarr = get_args(self.arg_spec, self.n)
        (old_dps, old_prec) = (mpmath.mp.dps, mpmath.mp.prec)
        try:
            if self.dps is not None:
                dps_list = [self.dps]
            else:
                dps_list = [20]
            if self.prec is not None:
                mpmath.mp.prec = self.prec
            if np.issubdtype(argarr.dtype, np.complexfloating):
                pytype = mpc2complex

                def mptype(x):
                    if False:
                        print('Hello World!')
                    return mpmath.mpc(complex(x))
            else:

                def mptype(x):
                    if False:
                        return 10
                    return mpmath.mpf(float(x))

                def pytype(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    if abs(x.imag) > 1e-16 * (1 + abs(x.real)):
                        return np.nan
                    else:
                        return mpf2float(x.real)
            for (j, dps) in enumerate(dps_list):
                mpmath.mp.dps = dps
                try:
                    assert_func_equal(self.scipy_func, lambda *a: pytype(self.mpmath_func(*map(mptype, a))), argarr, vectorized=False, rtol=self.rtol, atol=self.atol, ignore_inf_sign=self.ignore_inf_sign, distinguish_nan_and_inf=self.distinguish_nan_and_inf, nan_ok=self.nan_ok, param_filter=self.param_filter)
                    break
                except AssertionError:
                    if j >= len(dps_list) - 1:
                        (tp, value, tb) = sys.exc_info()
                        if value.__traceback__ is not tb:
                            raise value.with_traceback(tb)
                        raise value
        finally:
            (mpmath.mp.dps, mpmath.mp.prec) = (old_dps, old_prec)

    def __repr__(self):
        if False:
            return 10
        if self.is_complex:
            return f'<MpmathData: {self.name} (complex)>'
        else:
            return f'<MpmathData: {self.name}>'

def assert_mpmath_equal(*a, **kw):
    if False:
        return 10
    d = MpmathData(*a, **kw)
    d.check()

def nonfunctional_tooslow(func):
    if False:
        while True:
            i = 10
    return pytest.mark.skip(reason='    Test not yet functional (too slow), needs more work.')(func)

def mpf2float(x):
    if False:
        print('Hello World!')
    '\n    Convert an mpf to the nearest floating point number. Just using\n    float directly doesn\'t work because of results like this:\n\n    with mp.workdps(50):\n        float(mpf("0.99999999999999999")) = 0.9999999999999999\n\n    '
    return float(mpmath.nstr(x, 17, min_fixed=0, max_fixed=0))

def mpc2complex(x):
    if False:
        i = 10
        return i + 15
    return complex(mpf2float(x.real), mpf2float(x.imag))

def trace_args(func):
    if False:
        return 10

    def tofloat(x):
        if False:
            print('Hello World!')
        if isinstance(x, mpmath.mpc):
            return complex(x)
        else:
            return float(x)

    def wrap(*a, **kw):
        if False:
            for i in range(10):
                print('nop')
        sys.stderr.write(f'{tuple(map(tofloat, a))!r}: ')
        sys.stderr.flush()
        try:
            r = func(*a, **kw)
            sys.stderr.write('-> %r' % r)
        finally:
            sys.stderr.write('\n')
            sys.stderr.flush()
        return r
    return wrap
try:
    import signal
    POSIX = 'setitimer' in dir(signal)
except ImportError:
    POSIX = False

class TimeoutError(Exception):
    pass

def time_limited(timeout=0.5, return_val=np.nan, use_sigalrm=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator for setting a timeout for pure-Python functions.\n\n    If the function does not return within `timeout` seconds, the\n    value `return_val` is returned instead.\n\n    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is\n    used. Do not use this with threads: the SIGALRM implementation\n    does probably not work well. The settrace implementation only\n    traces the current thread.\n\n    The settrace implementation slows down execution speed. Slowdown\n    by a factor around 10 is probably typical.\n    '
    if POSIX and use_sigalrm:

        def sigalrm_handler(signum, frame):
            if False:
                print('Hello World!')
            raise TimeoutError()

        def deco(func):
            if False:
                while True:
                    i = 10

            def wrap(*a, **kw):
                if False:
                    return 10
                old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    return return_val
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrap
    else:

        def deco(func):
            if False:
                i = 10
                return i + 15

            def wrap(*a, **kw):
                if False:
                    i = 10
                    return i + 15
                start_time = time.time()

                def trace(frame, event, arg):
                    if False:
                        for i in range(10):
                            print('nop')
                    if time.time() - start_time > timeout:
                        raise TimeoutError()
                    return trace
                sys.settrace(trace)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    sys.settrace(None)
                    return return_val
                finally:
                    sys.settrace(None)
            return wrap
    return deco

def exception_to_nan(func):
    if False:
        return 10
    'Decorate function to return nan if it raises an exception'

    def wrap(*a, **kw):
        if False:
            print('Hello World!')
        try:
            return func(*a, **kw)
        except Exception:
            return np.nan
    return wrap

def inf_to_nan(func):
    if False:
        i = 10
        return i + 15
    'Decorate function to return nan if it returns inf'

    def wrap(*a, **kw):
        if False:
            i = 10
            return i + 15
        v = func(*a, **kw)
        if not np.isfinite(v):
            return np.nan
        return v
    return wrap

def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
    if False:
        return 10
    "\n    Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it\n    can be done to higher precision than double.\n    "
    failures = []
    for (k, (resval, stdval)) in enumerate(zip_longest(res, std)):
        if resval is None or stdval is None:
            raise ValueError('Lengths of inputs res and std are not equal.')
        if mpmath.fabs(resval - stdval) > atol + rtol * mpmath.fabs(stdval):
            failures.append((k, resval, stdval))
    nfail = len(failures)
    if nfail > 0:
        ndigits = int(abs(np.log10(rtol)))
        msg = ['']
        msg.append('Bad results ({} out of {}) for the following points:'.format(nfail, k + 1))
        for (k, resval, stdval) in failures:
            resrep = mpmath.nstr(resval, ndigits, min_fixed=0, max_fixed=0)
            stdrep = mpmath.nstr(stdval, ndigits, min_fixed=0, max_fixed=0)
            if stdval == 0:
                rdiff = 'inf'
            else:
                rdiff = mpmath.fabs((resval - stdval) / stdval)
                rdiff = mpmath.nstr(rdiff, 3)
            msg.append('{}: {} != {} (rdiff {})'.format(k, resrep, stdrep, rdiff))
        assert_(False, '\n'.join(msg))