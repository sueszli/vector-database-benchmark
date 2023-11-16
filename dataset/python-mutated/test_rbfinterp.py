import pickle
import pytest
import warnings
import numpy as _np
from numpy.linalg import LinAlgError
import cupy as cp
import cupyx
from cupy import testing
import cupyx.scipy.interpolate
try:
    from scipy import interpolate
except ImportError:
    pass
try:
    from scipy.stats.qmc import Halton
except ImportError:
    pass
from cupyx.scipy.interpolate._rbfinterp import _AVAILABLE, _SCALE_INVARIANT, _NAME_TO_MIN_DEGREE, NAME_TO_FUNC, _monomial_powers, polynomial_matrix, kernel_matrix

def _kernel_matrix(x, kernel):
    if False:
        print('Hello World!')
    'Return RBFs, with centers at `x`, evaluated at `x`.'
    out = cp.empty((x.shape[0], x.shape[0]), dtype=float)
    kernel_func = NAME_TO_FUNC[kernel]
    kernel_matrix(x, kernel_func, out)
    return out

def _polynomial_matrix(x, powers):
    if False:
        i = 10
        return i + 15
    'Return monomials, with exponents from `powers`, evaluated at `x`.'
    out = cp.empty((x.shape[0], powers.shape[0]), dtype=float)
    polynomial_matrix(x, powers, out)
    return out

def _vandermonde(x, degree):
    if False:
        print('Hello World!')
    powers = _monomial_powers(x.shape[1], degree)
    return _polynomial_matrix(x, powers)

def _1d_test_function(x, xp):
    if False:
        return 10
    x = x[:, 0]
    y = 4.26 * (xp.exp(-x) - 4 * xp.exp(-2 * x) + 3 * xp.exp(-3 * x))
    return y

def _2d_test_function(x, xp):
    if False:
        i = 10
        return i + 15
    (x1, x2) = (x[:, 0], x[:, 1])
    term1 = 0.75 * xp.exp(-(9 * x1 - 2) ** 2 / 4 - (9 * x2 - 2) ** 2 / 4)
    term2 = 0.75 * xp.exp(-(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) / 10)
    term3 = 0.5 * xp.exp(-(9 * x1 - 7) ** 2 / 4 - (9 * x2 - 3) ** 2 / 4)
    term4 = -0.2 * xp.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
    y = term1 + term2 + term3 + term4
    return y

def _is_conditionally_positive_definite(kernel, m, xp, scp):
    if False:
        print('Hello World!')
    nx = 10
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        seq = Halton(ndim, scramble=False, seed=_np.random.RandomState())
        for _ in range(ntests):
            x = xp.asarray(2 * seq.random(nx)) - 1
            A = _kernel_matrix(x, kernel)
            P = _vandermonde(x, m - 1)
            (Q, R) = cp.linalg.qr(P, mode='complete')
            Q2 = Q[:, P.shape[1]:]
            B = Q2.T.dot(A).dot(Q2)
            try:
                cp.linalg.cholesky(B)
            except cp.linalg.LinAlgError:
                return False
    return True

@testing.with_requires('scipy>=1.7.0')
@pytest.mark.skip(reason='conditionally posdef: skip for now')
@testing.numpy_cupy_allclose(scipy_name='scp')
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_conditionally_positive_definite(xp, scp, kernel):
    if False:
        while True:
            i = 10
    m = _NAME_TO_MIN_DEGREE.get(kernel, -1) + 1
    assert _is_conditionally_positive_definite(kernel, m, xp, scp)

@testing.with_requires('scipy>=1.7.0')
class _TestRBFInterpolator:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_1d(self, xp, scp, kernel):
        if False:
            i = 10
            return i + 15
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(3 * seq.random(50))
        y = _1d_test_function(x, xp)
        xitp = xp.asarray(3 * seq.random(50))
        yitp1 = self.build(scp, x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(scp, x, y, epsilon=2.0, kernel=kernel)(xitp)
        return (yitp1, yitp2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_2d(self, xp, scp, kernel):
        if False:
            return 10
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp)
        xitp = xp.asarray(seq.random(100))
        yitp1 = self.build(scp, x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(scp, x, y, epsilon=2.0, kernel=kernel)(xitp)
        return (yitp1, yitp2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_extreme_domains(self, xp, scp, kernel):
        if False:
            while True:
                i = 10
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        scale = 1e+50
        shift = 1e+55
        x = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp)
        xitp = xp.asarray(seq.random(100))
        if kernel in _SCALE_INVARIANT:
            yitp1 = self.build(scp, x, y, kernel=kernel)(xitp)
            yitp2 = self.build(scp, x * scale + shift, y, kernel=kernel)(xitp * scale + shift)
        else:
            yitp1 = self.build(scp, x, y, epsilon=5.0, kernel=kernel)(xitp)
            yitp2 = self.build(scp, x * scale + shift, y, epsilon=5.0 / scale, kernel=kernel)(xitp * scale + shift)
        return (yitp1, yitp2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_polynomial_reproduction(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        rng = _np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3
        x = xp.asarray(seq.random(50))
        xitp = xp.asarray(seq.random(50))
        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
            Pitp = _vandermonde(cp.asarray(xitp), degree).get()
        else:
            P = _vandermonde(x, degree)
            Pitp = _vandermonde(xitp, degree)
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        poly_coeffs = xp.asarray(poly_coeffs)
        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        yitp2 = self.build(scp, x, y, degree=degree)(xitp)
        return (yitp1, yitp2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vector_data(self, xp, scp):
        if False:
            print('Hello World!')
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))
        y = xp.array([_2d_test_function(x, xp), _2d_test_function(x[:, ::-1], xp)]).T
        yitp1 = self.build(scp, x, y)(xitp)
        yitp2 = self.build(scp, x, y[:, 0])(xitp)
        yitp3 = self.build(scp, x, y[:, 1])(xitp)
        return (yitp1, yitp2, yitp3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_data(self, xp, scp):
        if False:
            while True:
                i = 10
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp) + 1j * _2d_test_function(x[:, ::-1], xp)
        yitp1 = self.build(scp, x, y)(xitp)
        yitp2 = self.build(scp, x, y.real)(xitp)
        yitp3 = self.build(scp, x, y.imag)(xitp)
        return (yitp1, yitp2, yitp3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_1d(self, xp, scp, kernel):
        if False:
            for i in range(10):
                print('nop')
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(3 * seq.random(50))
        xitp = xp.asarray(3 * seq.random(50))
        y = _1d_test_function(x, xp)
        ytrue = _1d_test_function(xitp, xp)
        yitp = self.build(scp, x, y, epsilon=5.0, kernel=kernel)(xitp)
        mse = xp.mean((yitp - ytrue) ** 2)
        assert mse < 0.0001
        return yitp

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_2d(self, xp, scp, kernel):
        if False:
            return 10
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp)
        ytrue = _2d_test_function(xitp, xp)
        yitp = self.build(scp, x, y, epsilon=5.0, kernel=kernel)(xitp)
        mse = xp.mean((yitp - ytrue) ** 2)
        assert mse < 0.0002
        return yitp

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-08)
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_smoothing_misfit(self, xp, scp, kernel):
        if False:
            return 10
        rng = _np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        noise = 0.2
        rmse_tol = 0.1
        smoothing_range = 10 ** xp.linspace(-4, 1, 20)
        x = xp.asarray(3 * seq.random(100))
        y = _1d_test_function(x, xp) + xp.asarray(rng.normal(0.0, noise, (100,)))
        ytrue = _1d_test_function(x, xp)
        rmse_within_tol = False
        for smoothing in smoothing_range:
            ysmooth = self.build(scp, x, y, epsilon=1.0, smoothing=smoothing, kernel=kernel)(x)
            rmse = xp.sqrt(xp.mean((ysmooth - ytrue) ** 2))
            if rmse < rmse_tol:
                rmse_within_tol = True
                break
        assert rmse_within_tol
        return ysmooth

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_array_smoothing(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        rng = _np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        degree = 2
        x = xp.asarray(seq.random(50))
        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
        else:
            P = _vandermonde(x, degree)
        poly_coeffs = xp.asarray(rng.normal(0.0, 1.0, P.shape[1]))
        y = P.dot(poly_coeffs)
        y_with_outlier = xp.copy(y)
        y_with_outlier[10] += 1.0
        smoothing = xp.zeros((50,))
        smoothing[10] = 1000.0
        yitp = self.build(scp, x, y_with_outlier, smoothing=smoothing)(x)
        return (yitp, y)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_inconsistent_x_dimensions_error(self, xp, scp):
        if False:
            print('Hello World!')
        y = Halton(2, scramble=False, seed=_np.random.RandomState()).random(10)
        y = xp.asarray(y)
        d = _2d_test_function(y, xp)
        x = Halton(1, scramble=False, seed=_np.random.RandomState()).random(10)
        x = xp.asarray(x)
        self.build(scp, y, d)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_inconsistent_d_length_error(self, xp, scp):
        if False:
            print('Hello World!')
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(1)
        self.build(scp, y, d)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_y_not_2d_error(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        y = xp.linspace(0, 1, 5)
        d = xp.zeros(5)
        self.build(scp, y, d)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_inconsistent_smoothing_length_error(self, xp, scp):
        if False:
            while True:
                i = 10
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        smoothing = xp.ones(1)
        self.build(scp, y, d, smoothing=smoothing)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_invalid_kernel_name_error(self, xp, scp):
        if False:
            print('Hello World!')
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        self.build(scp, y, d, kernel='test')

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_epsilon_not_specified_error(self, xp, scp, kernel):
        if False:
            return 10
        if kernel in _SCALE_INVARIANT:
            return True
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        self.build(scp, y, d, kernel=kernel)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_x_not_2d_error(self, xp, scp):
        if False:
            return 10
        y = xp.linspace(0, 1, 5)[:, None]
        x = xp.linspace(0, 1, 5)
        d = xp.zeros(5)
        self.build(scp, y, d)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_not_enough_observations_error(self, xp, scp):
        if False:
            return 10
        y = xp.linspace(0, 1, 1)[:, None]
        d = xp.zeros(1)
        self.build(scp, y, d, kernel='thin_plate_spline')

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=UserWarning)
    @pytest.mark.parametrize('kernel', [kl for kl in _NAME_TO_MIN_DEGREE])
    def test_degree_warning(self, xp, scp, kernel):
        if False:
            print('Hello World!')
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        deg = _NAME_TO_MIN_DEGREE[kernel]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.build(scp, y, d, epsilon=1.0, kernel=kernel, degree=deg - 1)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=LinAlgError)
    def test_rank_error(self, xp, scp):
        if False:
            print('Hello World!')
        y = xp.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        d = xp.array([0.0, 0.0, 0.0])
        with cupyx.errstate(linalg='raise'):
            self.build(scp, y, d, kernel='thin_plate_spline')(y)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_single_point(self, xp, scp, dim):
        if False:
            i = 10
            return i + 15
        y = xp.zeros((1, dim))
        d = xp.ones((1,))
        f = self.build(scp, y, d, kernel='linear')(y)
        return (d, f)

    def test_pickleable(self):
        if False:
            return 10
        seq = Halton(1, scramble=False, seed=_np.random.RandomState(2305982309))
        x = cp.asarray(3 * seq.random(50))
        xitp = cp.asarray(3 * seq.random(50))
        y = _1d_test_function(x, cp)
        interp = cupyx.scipy.interpolate.RBFInterpolator(x, y)
        yitp1 = interp(xitp)
        yitp2 = pickle.loads(pickle.dumps(interp))(xitp)
        testing.assert_array_equal(yitp1, yitp2)

@testing.with_requires('scipy>=1.7.0')
class TestRBFInterpolatorNeighborsNone(_TestRBFInterpolator):

    def build(self, scp, *args, **kwargs):
        if False:
            while True:
                i = 10
        return scp.interpolate.RBFInterpolator(*args, **kwargs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_smoothing_limit_1d(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())
        degree = 3
        smoothing = 100000000.0
        x = xp.asarray(3 * seq.random(50))
        xitp = xp.asarray(3 * seq.random(50))
        y = _1d_test_function(x, xp)
        yitp1 = self.build(scp, x, y, degree=degree, smoothing=smoothing)(xitp)
        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
            Pitp = _vandermonde(cp.asarray(xitp), degree).get()
        else:
            P = _vandermonde(x, degree)
            Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(xp.linalg.lstsq(P, y, rcond=None)[0])
        return (yitp1, yitp2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_smoothing_limit_2d(self, xp, scp):
        if False:
            print('Hello World!')
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        degree = 3
        smoothing = 100000000.0
        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp)
        yitp1 = self.build(scp, x, y, degree=degree, smoothing=smoothing)(xitp)
        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
            Pitp = _vandermonde(cp.asarray(xitp), degree).get()
        else:
            P = _vandermonde(x, degree)
            Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(xp.linalg.lstsq(P, y, rcond=None)[0])
        return (yitp1, yitp2)

    @pytest.mark.slow
    def test_chunking(self):
        if False:
            return 10
        rng = _np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3
        largeN = 1000 + 33
        x = cp.asarray(seq.random(50))
        xitp = cp.asarray(seq.random(largeN))
        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        poly_coeffs = cp.asarray(rng.normal(0.0, 1.0, P.shape[1]))
        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        interp = cupyx.scipy.interpolate.RBFInterpolator(x, y, degree=degree)
        ce_real = interp._chunk_evaluator

        def _chunk_evaluator(*args, **kwargs):
            if False:
                print('Hello World!')
            kwargs.update(memory_budget=100)
            return ce_real(*args, **kwargs)
        interp._chunk_evaluator = _chunk_evaluator
        yitp2 = interp(xitp)
        testing.assert_allclose(yitp1, yitp2, atol=1e-08)
'\n# Disable `all neighbors not None` tests : they need KDTree\n\nclass TestRBFInterpolatorNeighbors20(_TestRBFInterpolator):\n    # RBFInterpolator using 20 nearest neighbors.\n    def build(self, *args, **kwargs):\n        return RBFInterpolator(*args, **kwargs, neighbors=20)\n\n    def test_equivalent_to_rbf_interpolator(self):\n        seq = Halton(2, scramble=False, seed=_np.random.RandomState())\n\n        x = cp.asarray(seq.random(100))\n        xitp = cp.asarray(seq.random(100))\n\n        y = _2d_test_function(x)\n\n        yitp1 = self.build(x, y)(xitp)\n\n        yitp2 = []\n        tree = cKDTree(x)\n        for xi in xitp:\n            _, nbr = tree.query(xi, 20)\n            yitp2.append(RBFInterpolator(x[nbr], y[nbr])(xi[None])[0])\n\n        assert_allclose(yitp1, yitp2, atol=1e-8)\n\n\nclass TestRBFInterpolatorNeighborsInf(TestRBFInterpolatorNeighborsNone):\n    # RBFInterpolator using neighbors=np.inf. This should give exactly the same\n    # results as neighbors=None, but it will be slower.\n    def build(self, *args, **kwargs):\n        return RBFInterpolator(*args, **kwargs, neighbors=cp.inf)\n\n    def test_equivalent_to_rbf_interpolator(self):\n        seq = Halton(1, scramble=False, seed=_np.random.RandomState())\n\n        x = cp.asarray(3*seq.random(50))\n        xitp = cp.asarray(3*seq.random(50))\n\n        y = _1d_test_function(x)\n        yitp1 = self.build(x, y)(xitp)\n        yitp2 = RBFInterpolator(x, y)(xitp)\n\n        assert_allclose(yitp1, yitp2, atol=1e-8)\n'