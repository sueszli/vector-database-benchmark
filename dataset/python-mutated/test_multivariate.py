"""
Test functions for multivariate normal distributions.

"""
import pickle
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_equal, assert_array_less, assert_
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import _PSD, _lnB, _cho_inv_batch, multivariate_normal_frozen
from scipy.stats import multivariate_normal, multivariate_hypergeom, matrix_normal, special_ortho_group, ortho_group, random_correlation, unitary_group, dirichlet, beta, wishart, multinomial, invwishart, chi2, invgamma, norm, uniform, ks_2samp, kstest, binom, hypergeom, multivariate_t, cauchy, normaltest, random_table, uniform_direction, vonmises_fisher, dirichlet_multinomial, vonmises
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch

def assert_close(res, ref, *args, **kwargs):
    if False:
        while True:
            i = 10
    (res, ref) = (np.asarray(res), np.asarray(ref))
    assert_allclose(res, ref, *args, **kwargs)
    assert_equal(res.shape, ref.shape)

class TestCovariance:

    def test_input_validation(self):
        if False:
            print('Hello World!')
        message = 'The input `precision` must be a square, two-dimensional...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaPrecision(np.ones(2))
        message = '`precision.shape` must equal `covariance.shape`.'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaPrecision(np.eye(3), covariance=np.eye(2))
        message = 'The input `diagonal` must be a one-dimensional array...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaDiagonal('alpaca')
        message = 'The input `cholesky` must be a square, two-dimensional...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaCholesky(np.ones(2))
        message = 'The input `eigenvalues` must be a one-dimensional...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition(('alpaca', np.eye(2)))
        message = 'The input `eigenvectors` must be a square...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition((np.ones(2), 'alpaca'))
        message = 'The shapes of `eigenvalues` and `eigenvectors` must be...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition(([1, 2, 3], np.eye(2)))
    _covariance_preprocessing = {'Diagonal': np.diag, 'Precision': np.linalg.inv, 'Cholesky': np.linalg.cholesky, 'Eigendecomposition': np.linalg.eigh, 'PSD': lambda x: _PSD(x, allow_singular=True)}
    _all_covariance_types = np.array(list(_covariance_preprocessing))
    _matrices = {'diagonal full rank': np.diag([1, 2, 3]), 'general full rank': [[5, 1, 3], [1, 6, 4], [3, 4, 7]], 'diagonal singular': np.diag([1, 0, 3]), 'general singular': [[5, -1, 0], [-1, 5, 0], [0, 0, 0]]}
    _cov_types = {'diagonal full rank': _all_covariance_types, 'general full rank': _all_covariance_types[1:], 'diagonal singular': _all_covariance_types[[0, -2, -1]], 'general singular': _all_covariance_types[-2:]}

    @pytest.mark.parametrize('cov_type_name', _all_covariance_types[:-1])
    def test_factories(self, cov_type_name):
        if False:
            return 10
        A = np.diag([1, 2, 3])
        x = [-4, 2, 5]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        factory = getattr(Covariance, f'from_{cov_type_name.lower()}')
        res = factory(preprocessing(A))
        ref = cov_type(preprocessing(A))
        assert type(res) == type(ref)
        assert_allclose(res.whiten(x), ref.whiten(x))

    @pytest.mark.parametrize('matrix_type', list(_matrices))
    @pytest.mark.parametrize('cov_type_name', _all_covariance_types)
    def test_covariance(self, matrix_type, cov_type_name):
        if False:
            for i in range(10):
                print('nop')
        message = f'CovVia{cov_type_name} does not support {matrix_type} matrices'
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        psd = _PSD(A, allow_singular=True)
        cov_object = cov_type(preprocessing(A))
        assert_close(cov_object.log_pdet, psd.log_pdet)
        assert_equal(cov_object.rank, psd.rank)
        assert_equal(cov_object.shape, np.asarray(A).shape)
        assert_close(cov_object.covariance, np.asarray(A))
        rng = np.random.default_rng(5292808890472453840)
        x = rng.random(size=3)
        res = cov_object.whiten(x)
        ref = x @ psd.U
        assert_close(res @ res, ref @ ref)
        if hasattr(cov_object, '_colorize') and 'singular' not in matrix_type:
            assert_close(cov_object.colorize(res), x)
        x = rng.random(size=(2, 4, 3))
        res = cov_object.whiten(x)
        ref = x @ psd.U
        assert_close((res ** 2).sum(axis=-1), (ref ** 2).sum(axis=-1))
        if hasattr(cov_object, '_colorize') and 'singular' not in matrix_type:
            assert_close(cov_object.colorize(res), x)
        if hasattr(cov_object, '_colorize'):
            res = cov_object.colorize(np.eye(len(A)))
            assert_close(res.T @ res, A)

    @pytest.mark.parametrize('size', [None, tuple(), 1, (2, 4, 3)])
    @pytest.mark.parametrize('matrix_type', list(_matrices))
    @pytest.mark.parametrize('cov_type_name', _all_covariance_types)
    def test_mvn_with_covariance(self, size, matrix_type, cov_type_name):
        if False:
            i = 10
            return i + 15
        message = f'CovVia{cov_type_name} does not support {matrix_type} matrices'
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        mean = [0.1, 0.2, 0.3]
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)
        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)
        rng = np.random.default_rng(5292808890472453840)
        x1 = mvn.rvs(mean, cov_object, size=size, random_state=rng)
        rng = np.random.default_rng(5292808890472453840)
        x2 = mvn(mean, cov_object, seed=rng).rvs(size=size)
        if isinstance(cov_object, _covariance.CovViaPSD):
            assert_close(x1, np.squeeze(x))
            assert_close(x2, np.squeeze(x))
        else:
            assert_equal(x1.shape, x.shape)
            assert_equal(x2.shape, x.shape)
            assert_close(x2, x1)
        assert_close(mvn.pdf(x, mean, cov_object), dist0.pdf(x))
        assert_close(dist1.pdf(x), dist0.pdf(x))
        assert_close(mvn.logpdf(x, mean, cov_object), dist0.logpdf(x))
        assert_close(dist1.logpdf(x), dist0.logpdf(x))
        assert_close(mvn.entropy(mean, cov_object), dist0.entropy())
        assert_close(dist1.entropy(), dist0.entropy())

    @pytest.mark.parametrize('size', [tuple(), (2, 4, 3)])
    @pytest.mark.parametrize('cov_type_name', _all_covariance_types)
    def test_mvn_with_covariance_cdf(self, size, cov_type_name):
        if False:
            i = 10
            return i + 15
        matrix_type = 'diagonal full rank'
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        mean = [0.1, 0.2, 0.3]
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)
        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)
        assert_close(mvn.cdf(x, mean, cov_object), dist0.cdf(x))
        assert_close(dist1.cdf(x), dist0.cdf(x))
        assert_close(mvn.logcdf(x, mean, cov_object), dist0.logcdf(x))
        assert_close(dist1.logcdf(x), dist0.logcdf(x))

    def test_covariance_instantiation(self):
        if False:
            i = 10
            return i + 15
        message = 'The `Covariance` class cannot be instantiated directly.'
        with pytest.raises(NotImplementedError, match=message):
            Covariance()

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_gh9942(self):
        if False:
            for i in range(10):
                print('nop')
        A = np.diag([1, 2, -1e-08])
        n = A.shape[0]
        mean = np.zeros(n)
        with pytest.raises(ValueError, match='The input matrix must be...'):
            multivariate_normal(mean, A).rvs()
        seed = 3562050283508273023
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        cov = Covariance.from_eigendecomposition(np.linalg.eigh(A))
        rv = multivariate_normal(mean, cov)
        res = rv.rvs(random_state=rng1)
        ref = multivariate_normal.rvs(mean, cov, random_state=rng2)
        assert_equal(res, ref)

    def test_gh19197(self):
        if False:
            return 10
        mean = np.ones(2)
        cov = Covariance.from_eigendecomposition((np.zeros(2), np.eye(2)))
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        rvs = dist.rvs(size=None)
        assert_equal(rvs, mean)
        cov = scipy.stats.Covariance.from_eigendecomposition((np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 400.0]])))
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        rvs = dist.rvs(size=None)
        assert rvs[0] != mean[0]
        assert rvs[1] == mean[1]

def _random_covariance(dim, evals, rng, singular=False):
    if False:
        for i in range(10):
            print('nop')
    A = rng.random((dim, dim))
    A = A @ A.T
    (_, v) = np.linalg.eigh(A)
    if singular:
        zero_eigs = rng.normal(size=dim) > 0
        evals[zero_eigs] = 0
    cov = v @ np.diag(evals) @ v.T
    return cov

def _sample_orthonormal_matrix(n):
    if False:
        return 10
    M = np.random.randn(n, n)
    (u, s, v) = scipy.linalg.svd(M)
    return u

class TestMultivariateNormal:

    def test_input_shape(self):
        if False:
            while True:
                i = 10
        mu = np.arange(3)
        cov = np.identity(2)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1, 2), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1, 2), mu, cov)

    def test_scalar_values(self):
        if False:
            return 10
        np.random.seed(1234)
        (x, mean, cov) = (1.5, 1.7, 2.5)
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)
        (x, mean, cov) = (1.5, 1.7, 2.5)
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)

    def test_logpdf(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logpdf(x, mean, cov)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    def test_logpdf_default_values(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logpdf(x)
        d2 = multivariate_normal.pdf(x)
        d3 = multivariate_normal.logpdf(x, None, 1)
        d4 = multivariate_normal.pdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    def test_logcdf(self):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logcdf(x, mean, cov)
        d2 = multivariate_normal.cdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    def test_logcdf_default_values(self):
        if False:
            return 10
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logcdf(x)
        d2 = multivariate_normal.cdf(x)
        d3 = multivariate_normal.logcdf(x, None, 1)
        d4 = multivariate_normal.cdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    def test_rank(self):
        if False:
            while True:
                i = 10
        np.random.seed(1234)
        n = 4
        mean = np.random.randn(n)
        for expected_rank in range(1, n + 1):
            s = np.random.randn(n, expected_rank)
            cov = np.dot(s, s.T)
            distn = multivariate_normal(mean, cov, allow_singular=True)
            assert_equal(distn.cov_object.rank, expected_rank)

    def test_degenerate_distributions(self):
        if False:
            return 10
        for n in range(1, 5):
            z = np.random.randn(n)
            for k in range(1, n):
                s = np.random.randn(k, k)
                cov_kk = np.dot(s, s.T)
                cov_nn = np.zeros((n, n))
                cov_nn[:k, :k] = cov_kk
                x = np.zeros(n)
                x[:k] = z[:k]
                u = _sample_orthonormal_matrix(n)
                cov_rr = np.dot(u, np.dot(cov_nn, u.T))
                y = np.dot(u, x)
                distn_kk = multivariate_normal(np.zeros(k), cov_kk, allow_singular=True)
                distn_nn = multivariate_normal(np.zeros(n), cov_nn, allow_singular=True)
                distn_rr = multivariate_normal(np.zeros(n), cov_rr, allow_singular=True)
                assert_equal(distn_kk.cov_object.rank, k)
                assert_equal(distn_nn.cov_object.rank, k)
                assert_equal(distn_rr.cov_object.rank, k)
                pdf_kk = distn_kk.pdf(x[:k])
                pdf_nn = distn_nn.pdf(x)
                pdf_rr = distn_rr.pdf(y)
                assert_allclose(pdf_kk, pdf_nn)
                assert_allclose(pdf_kk, pdf_rr)
                logpdf_kk = distn_kk.logpdf(x[:k])
                logpdf_nn = distn_nn.logpdf(x)
                logpdf_rr = distn_rr.logpdf(y)
                assert_allclose(logpdf_kk, logpdf_nn)
                assert_allclose(logpdf_kk, logpdf_rr)
                y_orth = y + u[:, -1]
                pdf_rr_orth = distn_rr.pdf(y_orth)
                logpdf_rr_orth = distn_rr.logpdf(y_orth)
                assert_equal(pdf_rr_orth, 0.0)
                assert_equal(logpdf_rr_orth, -np.inf)

    def test_degenerate_array(self):
        if False:
            for i in range(10):
                print('nop')
        k = 10
        for n in range(2, 6):
            for r in range(1, n):
                mn = np.zeros(n)
                u = _sample_orthonormal_matrix(n)[:, :r]
                vr = np.dot(u, u.T)
                X = multivariate_normal.rvs(mean=mn, cov=vr, size=k)
                pdf = multivariate_normal.pdf(X, mean=mn, cov=vr, allow_singular=True)
                assert_equal(pdf.size, k)
                assert np.all(pdf > 0.0)
                logpdf = multivariate_normal.logpdf(X, mean=mn, cov=vr, allow_singular=True)
                assert_equal(logpdf.size, k)
                assert np.all(logpdf > -np.inf)

    def test_large_pseudo_determinant(self):
        if False:
            return 10
        large_total_log = 1000.0
        npos = 100
        nzero = 2
        large_entry = np.exp(large_total_log / npos)
        n = npos + nzero
        cov = np.zeros((n, n), dtype=float)
        np.fill_diagonal(cov, large_entry)
        cov[-nzero:, -nzero:] = 0
        assert_equal(scipy.linalg.det(cov), 0)
        assert_equal(scipy.linalg.det(cov[:npos, :npos]), np.inf)
        assert_allclose(np.linalg.slogdet(cov[:npos, :npos]), (1, large_total_log))
        psd = _PSD(cov)
        assert_allclose(psd.log_pdet, large_total_log)

    def test_broadcasting(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        n = 4
        data = np.random.randn(n, n)
        cov = np.dot(data, data.T)
        mean = np.random.randn(n)
        X = np.random.randn(2, 3, n)
        desired_pdf = multivariate_normal.pdf(X, mean, cov)
        desired_cdf = multivariate_normal.cdf(X, mean, cov)
        for i in range(2):
            for j in range(3):
                actual = multivariate_normal.pdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_pdf[i, j])
                actual = multivariate_normal.cdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_cdf[i, j], rtol=0.001)

    def test_normal_1D(self):
        if False:
            while True:
                i = 10
        x = np.linspace(0, 2, 10)
        (mean, cov) = (1.2, 0.9)
        scale = cov ** 0.5
        d1 = norm.pdf(x, mean, scale)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, d2)
        d1 = norm.cdf(x, mean, scale)
        d2 = multivariate_normal.cdf(x, mean, cov)
        assert_allclose(d1, d2)

    def test_marginalization(self):
        if False:
            i = 10
            return i + 15
        mean = np.array([2.5, 3.5])
        cov = np.array([[0.5, 0.2], [0.2, 0.6]])
        n = 2 ** 8 + 1
        delta = 6 / (n - 1)
        v = np.linspace(0, 6, n)
        (xv, yv) = np.meshgrid(v, v)
        pos = np.empty((n, n, 2))
        pos[:, :, 0] = xv
        pos[:, :, 1] = yv
        pdf = multivariate_normal.pdf(pos, mean, cov)
        margin_x = romb(pdf, delta, axis=0)
        margin_y = romb(pdf, delta, axis=1)
        gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0] ** 0.5)
        gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1] ** 0.5)
        assert_allclose(margin_x, gauss_x, rtol=0.01, atol=0.01)
        assert_allclose(margin_y, gauss_y, rtol=0.01, atol=0.01)

    def test_frozen(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        norm_frozen = multivariate_normal(mean, cov)
        assert_allclose(norm_frozen.pdf(x), multivariate_normal.pdf(x, mean, cov))
        assert_allclose(norm_frozen.logpdf(x), multivariate_normal.logpdf(x, mean, cov))
        assert_allclose(norm_frozen.cdf(x), multivariate_normal.cdf(x, mean, cov))
        assert_allclose(norm_frozen.logcdf(x), multivariate_normal.logcdf(x, mean, cov))

    @pytest.mark.parametrize('covariance', [np.eye(2), Covariance.from_diagonal([1, 1])])
    def test_frozen_multivariate_normal_exposes_attributes(self, covariance):
        if False:
            print('Hello World!')
        mean = np.ones((2,))
        cov_should_be = np.eye(2)
        norm_frozen = multivariate_normal(mean, covariance)
        assert np.allclose(norm_frozen.mean, mean)
        assert np.allclose(norm_frozen.cov, cov_should_be)

    def test_pseudodet_pinv(self):
        if False:
            return 10
        np.random.seed(1234)
        n = 7
        x = np.random.randn(n, n)
        cov = np.dot(x, x.T)
        (s, u) = scipy.linalg.eigh(cov)
        s = np.full(n, 0.5)
        s[0] = 1.0
        s[-1] = 1e-07
        cov = np.dot(u, np.dot(np.diag(s), u.T))
        cond = 1e-05
        psd = _PSD(cov, cond=cond)
        psd_pinv = _PSD(psd.pinv, cond=cond)
        assert_allclose(psd.log_pdet, np.sum(np.log(s[:-1])))
        assert_allclose(-psd.log_pdet, psd_pinv.log_pdet)

    def test_exception_nonsquare_cov(self):
        if False:
            print('Hello World!')
        cov = [[1, 2, 3], [4, 5, 6]]
        assert_raises(ValueError, _PSD, cov)

    def test_exception_nonfinite_cov(self):
        if False:
            for i in range(10):
                print('nop')
        cov_nan = [[1, 0], [0, np.nan]]
        assert_raises(ValueError, _PSD, cov_nan)
        cov_inf = [[1, 0], [0, np.inf]]
        assert_raises(ValueError, _PSD, cov_inf)

    def test_exception_non_psd_cov(self):
        if False:
            print('Hello World!')
        cov = [[1, 0], [0, -1]]
        assert_raises(ValueError, _PSD, cov)

    def test_exception_singular_cov(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.ones((5, 5))
        e = np.linalg.LinAlgError
        assert_raises(e, multivariate_normal, mean, cov)
        assert_raises(e, multivariate_normal.pdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logpdf, x, mean, cov)
        assert_raises(e, multivariate_normal.cdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logcdf, x, mean, cov)
        cov = [[1.0, 0.0], [1.0, 1.0]]
        msg = 'When `allow_singular is False`, the input matrix'
        with pytest.raises(np.linalg.LinAlgError, match=msg):
            multivariate_normal(cov=cov)

    def test_R_values(self):
        if False:
            while True:
                i = 10
        r_pdf = np.array([0.0002214706, 0.0013819953, 0.0049138692, 0.010380305, 0.01402508])
        x = np.linspace(0, 2, 5)
        y = 3 * x - 2
        z = x + np.cos(y)
        r = np.array([x, y, z]).T
        mean = np.array([1, 3, 2], 'd')
        cov = np.array([[1, 2, 0], [2, 5, 0.5], [0, 0.5, 3]], 'd')
        pdf = multivariate_normal.pdf(r, mean, cov)
        assert_allclose(pdf, r_pdf, atol=1e-10)
        r_cdf = np.array([0.0017866215, 0.0267142892, 0.0857098761, 0.1063242573, 0.2501068509])
        cdf = multivariate_normal.cdf(r, mean, cov)
        assert_allclose(cdf, r_cdf, atol=2e-05)
        r_cdf2 = np.array([0.01262147, 0.05838989, 0.18389571, 0.40696599, 0.66470577])
        r2 = np.array([x, y]).T
        mean2 = np.array([1, 3], 'd')
        cov2 = np.array([[1, 2], [2, 5]], 'd')
        cdf2 = multivariate_normal.cdf(r2, mean2, cov2)
        assert_allclose(cdf2, r_cdf2, atol=1e-05)

    def test_multivariate_normal_rvs_zero_covariance(self):
        if False:
            while True:
                i = 10
        mean = np.zeros(2)
        covariance = np.zeros((2, 2))
        model = multivariate_normal(mean, covariance, allow_singular=True)
        sample = model.rvs()
        assert_equal(sample, [0, 0])

    def test_rvs_shape(self):
        if False:
            for i in range(10):
                print('nop')
        N = 300
        d = 4
        sample = multivariate_normal.rvs(mean=np.zeros(d), cov=1, size=N)
        assert_equal(sample.shape, (N, d))
        sample = multivariate_normal.rvs(mean=None, cov=np.array([[2, 0.1], [0.1, 1]]), size=N)
        assert_equal(sample.shape, (N, 2))
        u = multivariate_normal(mean=0, cov=1)
        sample = u.rvs(N)
        assert_equal(sample.shape, (N,))

    def test_large_sample(self):
        if False:
            print('Hello World!')
        np.random.seed(2846)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        size = 5000
        sample = multivariate_normal.rvs(mean, cov, size)
        assert_allclose(numpy.cov(sample.T), cov, rtol=0.1)
        assert_allclose(sample.mean(0), mean, rtol=0.1)

    def test_entropy(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2846)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        rv = multivariate_normal(mean, cov)
        assert_almost_equal(rv.entropy(), multivariate_normal.entropy(mean, cov))
        eigs = np.linalg.eig(cov)[0]
        desired = 1 / 2 * (n * (np.log(2 * np.pi) + 1) + np.sum(np.log(eigs)))
        assert_almost_equal(desired, rv.entropy())

    def test_lnB(self):
        if False:
            return 10
        alpha = np.array([1, 1, 1])
        desired = 0.5
        assert_almost_equal(np.exp(_lnB(alpha)), desired)

    def test_cdf_with_lower_limit_arrays(self):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(2408071309372769818)
        mean = [0, 0]
        cov = np.eye(2)
        a = rng.random((4, 3, 2)) * 6 - 3
        b = rng.random((4, 3, 2)) * 6 - 3
        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        cdf2a = multivariate_normal.cdf(b, mean, cov)
        cdf2b = multivariate_normal.cdf(a, mean, cov)
        ab1 = np.concatenate((a[..., 0:1], b[..., 1:2]), axis=-1)
        ab2 = np.concatenate((a[..., 1:2], b[..., 0:1]), axis=-1)
        cdf2ab1 = multivariate_normal.cdf(ab1, mean, cov)
        cdf2ab2 = multivariate_normal.cdf(ab2, mean, cov)
        cdf2 = cdf2a + cdf2b - cdf2ab1 - cdf2ab2
        assert_allclose(cdf1, cdf2)

    def test_cdf_with_lower_limit_consistency(self):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(2408071309372769818)
        mean = rng.random(3)
        cov = rng.random((3, 3))
        cov = cov @ cov.T
        a = rng.random((2, 3)) * 6 - 3
        b = rng.random((2, 3)) * 6 - 3
        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        cdf2 = multivariate_normal(mean, cov).cdf(b, lower_limit=a)
        cdf3 = np.exp(multivariate_normal.logcdf(b, mean, cov, lower_limit=a))
        cdf4 = np.exp(multivariate_normal(mean, cov).logcdf(b, lower_limit=a))
        assert_allclose(cdf2, cdf1, rtol=0.0001)
        assert_allclose(cdf3, cdf1, rtol=0.0001)
        assert_allclose(cdf4, cdf1, rtol=0.0001)

    def test_cdf_signs(self):
        if False:
            return 10
        mean = np.zeros(3)
        cov = np.eye(3)
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        expected_signs = np.array([1, -1, -1, 1])
        cdf = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        assert_allclose(cdf, cdf[0] * expected_signs)

    def test_mean_cov(self):
        if False:
            print('Hello World!')
        P = np.diag(1 / np.array([1, 2, 3]))
        cov_object = _covariance.CovViaPrecision(P)
        message = '`cov` represents a covariance matrix in 3 dimensions...'
        with pytest.raises(ValueError, match=message):
            multivariate_normal.entropy([0, 0], cov_object)
        with pytest.raises(ValueError, match=message):
            multivariate_normal([0, 0], cov_object)
        x = [0.5, 0.5, 0.5]
        ref = multivariate_normal.pdf(x, [0, 0, 0], cov_object)
        assert_equal(multivariate_normal.pdf(x, cov=cov_object), ref)
        ref = multivariate_normal.pdf(x, [1, 1, 1], cov_object)
        assert_equal(multivariate_normal.pdf(x, 1, cov=cov_object), ref)

    def test_fit_wrong_fit_data_shape(self):
        if False:
            return 10
        data = [1, 3]
        error_msg = '`x` must be two-dimensional.'
        with pytest.raises(ValueError, match=error_msg):
            multivariate_normal.fit(data)

    @pytest.mark.parametrize('dim', (3, 5))
    def test_fit_correctness(self, dim):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(4385269356937404)
        x = rng.random((100, dim))
        (mean_est, cov_est) = multivariate_normal.fit(x)
        (mean_ref, cov_ref) = (np.mean(x, axis=0), np.cov(x.T, ddof=0))
        assert_allclose(mean_est, mean_ref, atol=1e-15)
        assert_allclose(cov_est, cov_ref, rtol=1e-15)

    def test_fit_both_parameters_fixed(self):
        if False:
            return 10
        data = np.full((2, 1), 3)
        mean_fixed = 1.0
        cov_fixed = np.atleast_2d(1.0)
        (mean, cov) = multivariate_normal.fit(data, fix_mean=mean_fixed, fix_cov=cov_fixed)
        assert_equal(mean, mean_fixed)
        assert_equal(cov, cov_fixed)

    @pytest.mark.parametrize('fix_mean', [np.zeros((2, 2)), np.zeros((3,))])
    def test_fit_fix_mean_input_validation(self, fix_mean):
        if False:
            while True:
                i = 10
        msg = '`fix_mean` must be a one-dimensional array the same length as the dimensionality of the vectors `x`.'
        with pytest.raises(ValueError, match=msg):
            multivariate_normal.fit(np.eye(2), fix_mean=fix_mean)

    @pytest.mark.parametrize('fix_cov', [np.zeros((2,)), np.zeros((3, 2)), np.zeros((4, 4))])
    def test_fit_fix_cov_input_validation_dimension(self, fix_cov):
        if False:
            while True:
                i = 10
        msg = '`fix_cov` must be a two-dimensional square array of same side length as the dimensionality of the vectors `x`.'
        with pytest.raises(ValueError, match=msg):
            multivariate_normal.fit(np.eye(3), fix_cov=fix_cov)

    def test_fit_fix_cov_not_positive_semidefinite(self):
        if False:
            i = 10
            return i + 15
        error_msg = '`fix_cov` must be symmetric positive semidefinite.'
        with pytest.raises(ValueError, match=error_msg):
            fix_cov = np.array([[1.0, 0.0], [0.0, -1.0]])
            multivariate_normal.fit(np.eye(2), fix_cov=fix_cov)

    def test_fit_fix_mean(self):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(4385269356937404)
        loc = rng.random(3)
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100, random_state=rng)
        (mean_free, cov_free) = multivariate_normal.fit(samples)
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free, cov=cov_free).sum()
        (mean_fix, cov_fix) = multivariate_normal.fit(samples, fix_mean=loc)
        assert_equal(mean_fix, loc)
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_fix).sum()
        assert logp_fix < logp_free
        A = rng.random((3, 3))
        m = 1e-08 * np.dot(A, A.T)
        cov_perturbed = cov_fix + m
        logp_perturbed = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_perturbed).sum()
        assert logp_perturbed < logp_fix

    def test_fit_fix_cov(self):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(4385269356937404)
        loc = rng.random(3)
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100, random_state=rng)
        (mean_free, cov_free) = multivariate_normal.fit(samples)
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free, cov=cov_free).sum()
        (mean_fix, cov_fix) = multivariate_normal.fit(samples, fix_cov=cov)
        assert_equal(mean_fix, np.mean(samples, axis=0))
        assert_equal(cov_fix, cov)
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_fix).sum()
        assert logp_fix < logp_free
        mean_perturbed = mean_fix + 1e-08 * rng.random(3)
        logp_perturbed = multivariate_normal.logpdf(samples, mean=mean_perturbed, cov=cov_fix).sum()
        assert logp_perturbed < logp_fix

class TestMatrixNormal:

    def test_bad_input(self):
        if False:
            return 10
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        assert_raises(ValueError, matrix_normal, np.zeros((5, 4, 3)))
        assert_raises(ValueError, matrix_normal, M, np.zeros(10), V)
        assert_raises(ValueError, matrix_normal, M, U, np.zeros(10))
        assert_raises(ValueError, matrix_normal, M, U, U)
        assert_raises(ValueError, matrix_normal, M, V, V)
        assert_raises(ValueError, matrix_normal, M.T, U, V)
        e = np.linalg.LinAlgError
        assert_raises(e, matrix_normal.rvs, M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal.rvs, M, np.ones((num_rows, num_rows)), V)
        assert_raises(e, matrix_normal, M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal, M, np.ones((num_rows, num_rows)), V)

    def test_default_inputs(self):
        if False:
            while True:
                i = 10
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        Z = np.zeros((num_rows, num_cols))
        Zr = np.zeros((num_rows, 1))
        Zc = np.zeros((1, num_cols))
        Ir = np.identity(num_rows)
        Ic = np.identity(num_cols)
        I1 = np.identity(1)
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U, colcov=V).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U).shape, (num_rows, 1))
        assert_equal(matrix_normal.rvs(colcov=V).shape, (1, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, colcov=V).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U, colcov=V).shape, (num_rows, num_cols))
        assert_equal(matrix_normal(mean=M).rowcov, Ir)
        assert_equal(matrix_normal(mean=M).colcov, Ic)
        assert_equal(matrix_normal(rowcov=U).mean, Zr)
        assert_equal(matrix_normal(rowcov=U).colcov, I1)
        assert_equal(matrix_normal(colcov=V).mean, Zc)
        assert_equal(matrix_normal(colcov=V).rowcov, I1)
        assert_equal(matrix_normal(mean=M, rowcov=U).colcov, Ic)
        assert_equal(matrix_normal(mean=M, colcov=V).rowcov, Ir)
        assert_equal(matrix_normal(rowcov=U, colcov=V).mean, Z)

    def test_covariance_expansion(self):
        if False:
            print('Hello World!')
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        Uv = np.full(num_rows, 0.2)
        Us = 0.2
        Vv = np.full(num_cols, 0.1)
        Vs = 0.1
        Ir = np.identity(num_rows)
        Ic = np.identity(num_cols)
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).rowcov, 0.2 * Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).colcov, 0.1 * Ic)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).rowcov, 0.2 * Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).colcov, 0.1 * Ic)

    def test_frozen_matrix_normal(self):
        if False:
            return 10
        for i in range(1, 5):
            for j in range(1, 5):
                M = np.full((i, j), 0.3)
                U = 0.5 * np.identity(i) + np.full((i, i), 0.5)
                V = 0.7 * np.identity(j) + np.full((j, j), 0.3)
                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
                rvs1 = frozen.rvs(random_state=1234)
                rvs2 = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, random_state=1234)
                assert_equal(rvs1, rvs2)
                X = frozen.rvs(random_state=1234)
                pdf1 = frozen.pdf(X)
                pdf2 = matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(pdf1, pdf2)
                logpdf1 = frozen.logpdf(X)
                logpdf2 = matrix_normal.logpdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(logpdf1, logpdf2)

    def test_matches_multivariate(self):
        if False:
            print('Hello World!')
        for i in range(1, 5):
            for j in range(1, 5):
                M = np.full((i, j), 0.3)
                U = 0.5 * np.identity(i) + np.full((i, i), 0.5)
                V = 0.7 * np.identity(j) + np.full((j, j), 0.3)
                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
                X = frozen.rvs(random_state=1234)
                pdf1 = frozen.pdf(X)
                logpdf1 = frozen.logpdf(X)
                entropy1 = frozen.entropy()
                vecX = X.T.flatten()
                vecM = M.T.flatten()
                cov = np.kron(V, U)
                pdf2 = multivariate_normal.pdf(vecX, mean=vecM, cov=cov)
                logpdf2 = multivariate_normal.logpdf(vecX, mean=vecM, cov=cov)
                entropy2 = multivariate_normal.entropy(mean=vecM, cov=cov)
                assert_allclose(pdf1, pdf2, rtol=1e-10)
                assert_allclose(logpdf1, logpdf2, rtol=1e-10)
                assert_allclose(entropy1, entropy2)

    def test_array_input(self):
        if False:
            while True:
                i = 10
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        N = 10
        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
        X1 = frozen.rvs(size=N, random_state=1234)
        X2 = frozen.rvs(size=N, random_state=4321)
        X = np.concatenate((X1[np.newaxis, :, :, :], X2[np.newaxis, :, :, :]), axis=0)
        assert_equal(X.shape, (2, N, num_rows, num_cols))
        array_logpdf = frozen.logpdf(X)
        assert_equal(array_logpdf.shape, (2, N))
        for i in range(2):
            for j in range(N):
                separate_logpdf = matrix_normal.logpdf(X[i, j], mean=M, rowcov=U, colcov=V)
                assert_allclose(separate_logpdf, array_logpdf[i, j], 1e-10)

    def test_moments(self):
        if False:
            return 10
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        N = 1000
        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
        X = frozen.rvs(size=N, random_state=1234)
        sample_mean = np.mean(X, axis=0)
        assert_allclose(sample_mean, M, atol=0.1)
        sample_colcov = np.cov(X.reshape(N * num_rows, num_cols).T)
        assert_allclose(sample_colcov, V, atol=0.1)
        sample_rowcov = np.cov(np.swapaxes(X, 1, 2).reshape(N * num_cols, num_rows).T)
        assert_allclose(sample_rowcov, U, atol=0.1)

    def test_samples(self):
        if False:
            while True:
                i = 10
        actual = matrix_normal.rvs(mean=np.array([[1, 2], [3, 4]]), rowcov=np.array([[4, -1], [-1, 2]]), colcov=np.array([[5, 1], [1, 10]]), random_state=np.random.default_rng(0), size=2)
        expected = np.array([[[1.56228264238181, -1.24136424071189], [2.46865788392114, 6.22964440489445]], [[3.86405716144353, 10.73714311429529], [2.59428444080606, 5.79987854490876]]])
        assert_allclose(actual, expected)

class TestDirichlet:

    def test_frozen_dirichlet(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2846)
        n = np.random.randint(1, 32)
        alpha = np.random.uniform(1e-09, 100, n)
        d = dirichlet(alpha)
        assert_equal(d.var(), dirichlet.var(alpha))
        assert_equal(d.mean(), dirichlet.mean(alpha))
        assert_equal(d.entropy(), dirichlet.entropy(alpha))
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(1e-09, 100, n)
            x /= np.sum(x)
            assert_equal(d.pdf(x[:-1]), dirichlet.pdf(x[:-1], alpha))
            assert_equal(d.logpdf(x[:-1]), dirichlet.logpdf(x[:-1], alpha))

    def test_numpy_rvs_shape_compatibility(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(2846)
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.random.dirichlet(alpha, size=7)
        assert_equal(x.shape, (7, 3))
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)
        dirichlet.pdf(x.T, alpha)
        dirichlet.pdf(x.T[:-1], alpha)
        dirichlet.logpdf(x.T, alpha)
        dirichlet.logpdf(x.T[:-1], alpha)

    def test_alpha_with_zeros(self):
        if False:
            print('Hello World!')
        np.random.seed(2846)
        alpha = [1.0, 0.0, 3.0]
        x = np.random.dirichlet(np.maximum(1e-09, alpha), size=7).T
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_with_negative_entries(self):
        if False:
            while True:
                i = 10
        np.random.seed(2846)
        alpha = [1.0, -2.0, 3.0]
        x = np.random.dirichlet(np.maximum(1e-09, alpha), size=7).T
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_zeros(self):
        if False:
            return 10
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)
        alpha = np.array([1.0, 1.0, 1.0, 1.0])
        assert_almost_equal(dirichlet.pdf(x, alpha), 6)
        assert_almost_equal(dirichlet.logpdf(x, alpha), np.log(6))

    def test_data_with_zeros_and_small_alpha(self):
        if False:
            for i in range(10):
                print('nop')
        alpha = np.array([1.0, 0.5, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_negative_entries(self):
        if False:
            while True:
                i = 10
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, -0.1, 0.3, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_too_large_entries(self):
        if False:
            print('Hello World!')
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 1.1, 0.3, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_too_deep_c(self):
        if False:
            for i in range(10):
                print('nop')
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((2, 7, 7), 1 / 14)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_too_deep(self):
        if False:
            while True:
                i = 10
        alpha = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.full((2, 2, 7), 1 / 4)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_correct_depth(self):
        if False:
            i = 10
            return i + 15
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((3, 7), 1 / 3)
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)

    def test_non_simplex_data(self):
        if False:
            print('Hello World!')
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((3, 7), 1 / 2)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_vector_too_short(self):
        if False:
            i = 10
            return i + 15
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.full((2, 7), 1 / 2)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_vector_too_long(self):
        if False:
            for i in range(10):
                print('nop')
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.full((5, 7), 1 / 5)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_mean_var_cov(self):
        if False:
            print('Hello World!')
        alpha = np.array([1.0, 0.8, 0.2])
        d = dirichlet(alpha)
        expected_mean = [0.5, 0.4, 0.1]
        expected_var = [1.0 / 12.0, 0.08, 0.03]
        expected_cov = [[1.0 / 12, -1.0 / 15, -1.0 / 60], [-1.0 / 15, 2.0 / 25, -1.0 / 75], [-1.0 / 60, -1.0 / 75, 3.0 / 100]]
        assert_array_almost_equal(d.mean(), expected_mean)
        assert_array_almost_equal(d.var(), expected_var)
        assert_array_almost_equal(d.cov(), expected_cov)

    def test_scalar_values(self):
        if False:
            while True:
                i = 10
        alpha = np.array([0.2])
        d = dirichlet(alpha)
        assert_equal(d.mean().ndim, 0)
        assert_equal(d.var().ndim, 0)
        assert_equal(d.pdf([1.0]).ndim, 0)
        assert_equal(d.logpdf([1.0]).ndim, 0)

    def test_K_and_K_minus_1_calls_equal(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2846)
        n = np.random.randint(1, 32)
        alpha = np.random.uniform(1e-09, 100, n)
        d = dirichlet(alpha)
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(1e-09, 100, n)
            x /= np.sum(x)
            assert_almost_equal(d.pdf(x[:-1]), d.pdf(x))

    def test_multiple_entry_calls(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2846)
        n = np.random.randint(1, 32)
        alpha = np.random.uniform(1e-09, 100, n)
        d = dirichlet(alpha)
        num_tests = 10
        num_multiple = 5
        xm = None
        for i in range(num_tests):
            for m in range(num_multiple):
                x = np.random.uniform(1e-09, 100, n)
                x /= np.sum(x)
                if xm is not None:
                    xm = np.vstack((xm, x))
                else:
                    xm = x
            rm = d.pdf(xm.T)
            rs = None
            for xs in xm:
                r = d.pdf(xs)
                if rs is not None:
                    rs = np.append(rs, r)
                else:
                    rs = r
            assert_array_almost_equal(rm, rs)

    def test_2D_dirichlet_is_beta(self):
        if False:
            return 10
        np.random.seed(2846)
        alpha = np.random.uniform(1e-09, 100, 2)
        d = dirichlet(alpha)
        b = beta(alpha[0], alpha[1])
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(1e-09, 100, 2)
            x /= np.sum(x)
            assert_almost_equal(b.pdf(x), d.pdf([x]))
        assert_almost_equal(b.mean(), d.mean()[0])
        assert_almost_equal(b.var(), d.var()[0])

def test_multivariate_normal_dimensions_mismatch():
    if False:
        i = 10
        return i + 15
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0]])
    assert_raises(ValueError, multivariate_normal, mu, sigma)
    try:
        multivariate_normal(mu, sigma)
    except ValueError as e:
        msg = 'Dimension mismatch'
        assert_equal(str(e)[:len(msg)], msg)

class TestWishart:

    def test_scale_dimensions(self):
        if False:
            print('Hello World!')
        true_scale = np.array(1, ndmin=2)
        scales = [1, [1], np.array(1), np.r_[1], np.array(1, ndmin=2)]
        for scale in scales:
            w = wishart(1, scale)
            assert_equal(w.scale, true_scale)
            assert_equal(w.scale.shape, true_scale.shape)
        true_scale = np.array([[1, 0], [0, 2]])
        scales = [[1, 2], np.r_[1, 2], np.array([[1, 0], [0, 2]])]
        for scale in scales:
            w = wishart(2, scale)
            assert_equal(w.scale, true_scale)
            assert_equal(w.scale.shape, true_scale.shape)
        assert_raises(ValueError, wishart, 1, np.eye(2))
        wishart(1.1, np.eye(2))
        scale = np.array(1, ndmin=3)
        assert_raises(ValueError, wishart, 1, scale)

    def test_quantile_dimensions(self):
        if False:
            while True:
                i = 10
        X = [1, [1], np.array(1), np.r_[1], np.array(1, ndmin=2), np.array([1], ndmin=3)]
        w = wishart(1, 1)
        density = w.pdf(np.array(1, ndmin=3))
        for x in X:
            assert_equal(w.pdf(x), density)
        X = [[1, 2, 3], np.r_[1, 2, 3], np.array([1, 2, 3], ndmin=3)]
        w = wishart(1, 1)
        density = w.pdf(np.array([1, 2, 3], ndmin=3))
        for x in X:
            assert_equal(w.pdf(x), density)
        X = [2, [2, 2], np.array(2), np.r_[2, 2], np.array([[2, 0], [0, 2]]), np.array([[2, 0], [0, 2]])[:, :, np.newaxis]]
        w = wishart(2, np.eye(2))
        density = w.pdf(np.array([[2, 0], [0, 2]])[:, :, np.newaxis])
        for x in X:
            assert_equal(w.pdf(x), density)

    def test_frozen(self):
        if False:
            for i in range(10):
                print('nop')
        dim = 4
        scale = np.diag(np.arange(dim) + 1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) // 2)
        scale = np.dot(scale.T, scale)
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim) + (i + 1) ** 2)
            x[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) // 2)
            x = np.dot(x.T, x)
            X.append(x)
        X = np.array(X).T
        parameters = [(10, 1, np.linspace(0.1, 10, 5)), (10, scale, X)]
        for (df, scale, x) in parameters:
            w = wishart(df, scale)
            assert_equal(w.var(), wishart.var(df, scale))
            assert_equal(w.mean(), wishart.mean(df, scale))
            assert_equal(w.mode(), wishart.mode(df, scale))
            assert_equal(w.entropy(), wishart.entropy(df, scale))
            assert_equal(w.pdf(x), wishart.pdf(x, df, scale))

    def test_1D_is_chisquared(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(482974)
        sn = 500
        dim = 1
        scale = np.eye(dim)
        df_range = np.arange(1, 10, 2, dtype=float)
        X = np.linspace(0.1, 10, num=10)
        for df in df_range:
            w = wishart(df, scale)
            c = chi2(df)
            assert_allclose(w.var(), c.var())
            assert_allclose(w.mean(), c.mean())
            assert_allclose(w.entropy(), c.entropy())
            assert_allclose(w.pdf(X), c.pdf(X))
            rvs = w.rvs(size=sn)
            args = (df,)
            alpha = 0.01
            check_distribution_rvs('chi2', args, alpha, rvs)

    def test_is_scaled_chisquared(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(482974)
        sn = 500
        df = 10
        dim = 4
        scale = np.diag(np.arange(4) + 1)
        scale[np.tril_indices(4, k=-1)] = np.arange(6)
        scale = np.dot(scale.T, scale)
        lamda = np.ones((dim, 1))
        sigma_lamda = lamda.T.dot(scale).dot(lamda).squeeze()
        w = wishart(df, sigma_lamda)
        c = chi2(df, scale=sigma_lamda)
        assert_allclose(w.var(), c.var())
        assert_allclose(w.mean(), c.mean())
        assert_allclose(w.entropy(), c.entropy())
        X = np.linspace(0.1, 10, num=10)
        assert_allclose(w.pdf(X), c.pdf(X))
        rvs = w.rvs(size=sn)
        args = (df, 0, sigma_lamda)
        alpha = 0.01
        check_distribution_rvs('chi2', args, alpha, rvs)

class TestMultinomial:

    def test_logpmf(self):
        if False:
            i = 10
            return i + 15
        vals1 = multinomial.logpmf((3, 4), 7, (0.3, 0.7))
        assert_allclose(vals1, -1.483270127243324, rtol=1e-08)
        vals2 = multinomial.logpmf([3, 4], 0, [0.3, 0.7])
        assert vals2 == -np.inf
        vals3 = multinomial.logpmf([0, 0], 0, [0.3, 0.7])
        assert vals3 == 0
        vals4 = multinomial.logpmf([3, 4], 0, [-2, 3])
        assert_allclose(vals4, np.nan, rtol=1e-08)

    def test_reduces_binomial(self):
        if False:
            print('Hello World!')
        val1 = multinomial.logpmf((3, 4), 7, (0.3, 0.7))
        val2 = binom.logpmf(3, 7, 0.3)
        assert_allclose(val1, val2, rtol=1e-08)
        val1 = multinomial.pmf((6, 8), 14, (0.1, 0.9))
        val2 = binom.pmf(6, 14, 0.1)
        assert_allclose(val1, val2, rtol=1e-08)

    def test_R(self):
        if False:
            for i in range(10):
                print('nop')
        (n, p) = (3, [1.0 / 8, 2.0 / 8, 5.0 / 8])
        r_vals = {(0, 0, 3): 0.244140625, (1, 0, 2): 0.146484375, (2, 0, 1): 0.029296875, (3, 0, 0): 0.001953125, (0, 1, 2): 0.29296875, (1, 1, 1): 0.1171875, (2, 1, 0): 0.01171875, (0, 2, 1): 0.1171875, (1, 2, 0): 0.0234375, (0, 3, 0): 0.015625}
        for x in r_vals:
            assert_allclose(multinomial.pmf(x, n, p), r_vals[x], atol=1e-14)

    @pytest.mark.parametrize('n', [0, 3])
    def test_rvs_np(self, n):
        if False:
            print('Hello World!')
        sc_rvs = multinomial.rvs(n, [1 / 4.0] * 3, size=7, random_state=123)
        rndm = np.random.RandomState(123)
        np_rvs = rndm.multinomial(n, [1 / 4.0] * 3, size=7)
        assert_equal(sc_rvs, np_rvs)

    def test_pmf(self):
        if False:
            while True:
                i = 10
        vals0 = multinomial.pmf((5,), 5, (1,))
        assert_allclose(vals0, 1, rtol=1e-08)
        vals1 = multinomial.pmf((3, 4), 7, (0.3, 0.7))
        assert_allclose(vals1, 0.22689449999999994, rtol=1e-08)
        vals2 = multinomial.pmf([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]], 8, (0.1, 0.9))
        assert_allclose(vals2, [[0.03306744, 0.43046721], [0, 0]], rtol=1e-08)
        x = np.empty((0, 2), dtype=np.float64)
        vals3 = multinomial.pmf(x, 4, (0.3, 0.7))
        assert_equal(vals3, np.empty([], dtype=np.float64))
        vals4 = multinomial.pmf([1, 2], 4, (0.3, 0.7))
        assert_allclose(vals4, 0, rtol=1e-08)
        vals5 = multinomial.pmf([3, 3, 0], 6, [2 / 3.0, 1 / 3.0, 0])
        assert_allclose(vals5, 0.219478737997, rtol=1e-08)
        vals5 = multinomial.pmf([0, 0, 0], 0, [2 / 3.0, 1 / 3.0, 0])
        assert vals5 == 1
        vals6 = multinomial.pmf([2, 1, 0], 0, [2 / 3.0, 1 / 3.0, 0])
        assert vals6 == 0

    def test_pmf_broadcasting(self):
        if False:
            while True:
                i = 10
        vals0 = multinomial.pmf([1, 2], 3, [[0.1, 0.9], [0.2, 0.8]])
        assert_allclose(vals0, [0.243, 0.384], rtol=1e-08)
        vals1 = multinomial.pmf([1, 2], [3, 4], [0.1, 0.9])
        assert_allclose(vals1, [0.243, 0], rtol=1e-08)
        vals2 = multinomial.pmf([[[1, 2], [1, 1]]], 3, [0.1, 0.9])
        assert_allclose(vals2, [[0.243, 0]], rtol=1e-08)
        vals3 = multinomial.pmf([1, 2], [[[3], [4]]], [0.1, 0.9])
        assert_allclose(vals3, [[[0.243], [0]]], rtol=1e-08)
        vals4 = multinomial.pmf([[1, 2], [1, 1]], [[[[3]]]], [0.1, 0.9])
        assert_allclose(vals4, [[[[0.243, 0]]]], rtol=1e-08)

    @pytest.mark.parametrize('n', [0, 5])
    def test_cov(self, n):
        if False:
            print('Hello World!')
        cov1 = multinomial.cov(n, (0.2, 0.3, 0.5))
        cov2 = [[n * 0.2 * 0.8, -n * 0.2 * 0.3, -n * 0.2 * 0.5], [-n * 0.3 * 0.2, n * 0.3 * 0.7, -n * 0.3 * 0.5], [-n * 0.5 * 0.2, -n * 0.5 * 0.3, n * 0.5 * 0.5]]
        assert_allclose(cov1, cov2, rtol=1e-08)

    def test_cov_broadcasting(self):
        if False:
            return 10
        cov1 = multinomial.cov(5, [[0.1, 0.9], [0.2, 0.8]])
        cov2 = [[[0.45, -0.45], [-0.45, 0.45]], [[0.8, -0.8], [-0.8, 0.8]]]
        assert_allclose(cov1, cov2, rtol=1e-08)
        cov3 = multinomial.cov([4, 5], [0.1, 0.9])
        cov4 = [[[0.36, -0.36], [-0.36, 0.36]], [[0.45, -0.45], [-0.45, 0.45]]]
        assert_allclose(cov3, cov4, rtol=1e-08)
        cov5 = multinomial.cov([4, 5], [[0.3, 0.7], [0.4, 0.6]])
        cov6 = [[[4 * 0.3 * 0.7, -4 * 0.3 * 0.7], [-4 * 0.3 * 0.7, 4 * 0.3 * 0.7]], [[5 * 0.4 * 0.6, -5 * 0.4 * 0.6], [-5 * 0.4 * 0.6, 5 * 0.4 * 0.6]]]
        assert_allclose(cov5, cov6, rtol=1e-08)

    @pytest.mark.parametrize('n', [0, 2])
    def test_entropy(self, n):
        if False:
            for i in range(10):
                print('nop')
        ent0 = multinomial.entropy(n, [0.2, 0.8])
        assert_allclose(ent0, binom.entropy(n, 0.2), rtol=1e-08)

    def test_entropy_broadcasting(self):
        if False:
            return 10
        ent0 = multinomial.entropy([2, 3], [0.2, 0.3])
        assert_allclose(ent0, [binom.entropy(2, 0.2), binom.entropy(3, 0.2)], rtol=1e-08)
        ent1 = multinomial.entropy([7, 8], [[0.3, 0.7], [0.4, 0.6]])
        assert_allclose(ent1, [binom.entropy(7, 0.3), binom.entropy(8, 0.4)], rtol=1e-08)
        ent2 = multinomial.entropy([[7], [8]], [[0.3, 0.7], [0.4, 0.6]])
        assert_allclose(ent2, [[binom.entropy(7, 0.3), binom.entropy(7, 0.4)], [binom.entropy(8, 0.3), binom.entropy(8, 0.4)]], rtol=1e-08)

    @pytest.mark.parametrize('n', [0, 5])
    def test_mean(self, n):
        if False:
            while True:
                i = 10
        mean1 = multinomial.mean(n, [0.2, 0.8])
        assert_allclose(mean1, [n * 0.2, n * 0.8], rtol=1e-08)

    def test_mean_broadcasting(self):
        if False:
            return 10
        mean1 = multinomial.mean([5, 6], [0.2, 0.8])
        assert_allclose(mean1, [[5 * 0.2, 5 * 0.8], [6 * 0.2, 6 * 0.8]], rtol=1e-08)

    def test_frozen(self):
        if False:
            return 10
        np.random.seed(1234)
        n = 12
        pvals = (0.1, 0.2, 0.3, 0.4)
        x = [[0, 0, 0, 12], [0, 0, 1, 11], [0, 1, 1, 10], [1, 1, 1, 9], [1, 1, 2, 8]]
        x = np.asarray(x, dtype=np.float64)
        mn_frozen = multinomial(n, pvals)
        assert_allclose(mn_frozen.pmf(x), multinomial.pmf(x, n, pvals))
        assert_allclose(mn_frozen.logpmf(x), multinomial.logpmf(x, n, pvals))
        assert_allclose(mn_frozen.entropy(), multinomial.entropy(n, pvals))

    def test_gh_11860(self):
        if False:
            return 10
        n = 88
        rng = np.random.default_rng(8879715917488330089)
        p = rng.random(n)
        p[-1] = 1e-30
        p /= np.sum(p)
        x = np.ones(n)
        logpmf = multinomial.logpmf(x, n, p)
        assert np.isfinite(logpmf)

class TestInvwishart:

    def test_frozen(self):
        if False:
            return 10
        dim = 4
        scale = np.diag(np.arange(dim) + 1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
        scale = np.dot(scale.T, scale)
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim) + (i + 1) ** 2)
            x[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
            x = np.dot(x.T, x)
            X.append(x)
        X = np.array(X).T
        parameters = [(10, 1, np.linspace(0.1, 10, 5)), (10, scale, X)]
        for (df, scale, x) in parameters:
            iw = invwishart(df, scale)
            assert_equal(iw.var(), invwishart.var(df, scale))
            assert_equal(iw.mean(), invwishart.mean(df, scale))
            assert_equal(iw.mode(), invwishart.mode(df, scale))
            assert_allclose(iw.pdf(x), invwishart.pdf(x, df, scale))

    def test_1D_is_invgamma(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(482974)
        sn = 500
        dim = 1
        scale = np.eye(dim)
        df_range = np.arange(5, 20, 2, dtype=float)
        X = np.linspace(0.1, 10, num=10)
        for df in df_range:
            iw = invwishart(df, scale)
            ig = invgamma(df / 2, scale=1.0 / 2)
            assert_allclose(iw.var(), ig.var())
            assert_allclose(iw.mean(), ig.mean())
            assert_allclose(iw.pdf(X), ig.pdf(X))
            rvs = iw.rvs(size=sn)
            args = (df / 2, 0, 1.0 / 2)
            alpha = 0.01
            check_distribution_rvs('invgamma', args, alpha, rvs)
            assert_allclose(iw.entropy(), ig.entropy())

    def test_wishart_invwishart_2D_rvs(self):
        if False:
            return 10
        dim = 3
        df = 10
        scale = np.eye(dim)
        scale[0, 1] = 0.5
        scale[1, 0] = 0.5
        w = wishart(df, scale)
        iw = invwishart(df, scale)
        np.random.seed(248042)
        w_rvs = wishart.rvs(df, scale)
        np.random.seed(248042)
        frozen_w_rvs = w.rvs()
        np.random.seed(248042)
        iw_rvs = invwishart.rvs(df, scale)
        np.random.seed(248042)
        frozen_iw_rvs = iw.rvs()
        np.random.seed(248042)
        covariances = np.random.normal(size=3)
        variances = np.r_[np.random.chisquare(df), np.random.chisquare(df - 1), np.random.chisquare(df - 2)] ** 0.5
        A = np.diag(variances)
        A[np.tril_indices(dim, k=-1)] = covariances
        D = np.linalg.cholesky(scale)
        DA = D.dot(A)
        manual_w_rvs = np.dot(DA, DA.T)
        iD = np.linalg.cholesky(np.linalg.inv(scale))
        iDA = iD.dot(A)
        manual_iw_rvs = np.linalg.inv(np.dot(iDA, iDA.T))
        assert_allclose(w_rvs, manual_w_rvs)
        assert_allclose(frozen_w_rvs, manual_w_rvs)
        assert_allclose(iw_rvs, manual_iw_rvs)
        assert_allclose(frozen_iw_rvs, manual_iw_rvs)

    def test_cho_inv_batch(self):
        if False:
            while True:
                i = 10
        'Regression test for gh-8844.'
        a0 = np.array([[2, 1, 0, 0.5], [1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 2]])
        a1 = np.array([[2, -1, 0, 0.5], [-1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 4]])
        a = np.array([a0, a1])
        ainv = a.copy()
        _cho_inv_batch(ainv)
        ident = np.eye(4)
        assert_allclose(a[0].dot(ainv[0]), ident, atol=1e-15)
        assert_allclose(a[1].dot(ainv[1]), ident, atol=1e-15)

    def test_logpdf_4x4(self):
        if False:
            return 10
        'Regression test for gh-8844.'
        X = np.array([[2, 1, 0, 0.5], [1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 2]])
        Psi = np.array([[9, 7, 3, 1], [7, 9, 5, 1], [3, 5, 8, 2], [1, 1, 2, 9]])
        nu = 6
        prob = invwishart.logpdf(X, nu, Psi)
        p = X.shape[0]
        (sig, logdetX) = np.linalg.slogdet(X)
        (sig, logdetPsi) = np.linalg.slogdet(Psi)
        M = np.linalg.solve(X, Psi)
        expected = nu / 2 * logdetPsi - nu * p / 2 * np.log(2) - multigammaln(nu / 2, p) - (nu + p + 1) / 2 * logdetX - 0.5 * M.trace()
        assert_allclose(prob, expected)

class TestSpecialOrthoGroup:

    def test_reproducibility(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(514)
        x = special_ortho_group.rvs(3)
        expected = np.array([[-0.99394515, -0.04527879, 0.10011432], [0.04821555, -0.99846897, 0.02711042], [0.09873351, 0.03177334, 0.99460653]])
        assert_array_almost_equal(x, expected)
        random_state = np.random.RandomState(seed=514)
        x = special_ortho_group.rvs(3, random_state=random_state)
        assert_array_almost_equal(x, expected)

    def test_invalid_dim(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, special_ortho_group.rvs, None)
        assert_raises(ValueError, special_ortho_group.rvs, (2, 2))
        assert_raises(ValueError, special_ortho_group.rvs, 1)
        assert_raises(ValueError, special_ortho_group.rvs, 2.5)

    def test_frozen_matrix(self):
        if False:
            i = 10
            return i + 15
        dim = 7
        frozen = special_ortho_group(dim)
        rvs1 = frozen.rvs(random_state=1234)
        rvs2 = special_ortho_group.rvs(dim, random_state=1234)
        assert_equal(rvs1, rvs2)

    def test_det_and_ortho(self):
        if False:
            for i in range(10):
                print('nop')
        xs = [special_ortho_group.rvs(dim) for dim in range(2, 12) for i in range(3)]
        dets = [np.linalg.det(x) for x in xs]
        assert_allclose(dets, [1.0] * 30, rtol=1e-13)
        for x in xs:
            assert_array_almost_equal(np.dot(x, x.T), np.eye(x.shape[0]))

    def test_haar(self):
        if False:
            print('Hello World!')
        dim = 5
        samples = 1000
        ks_prob = 0.05
        np.random.seed(514)
        xs = special_ortho_group.rvs(dim, size=samples)
        els = ((0, 0), (0, 2), (1, 4), (2, 3))
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for (er, ec) in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob] * len(pairs), ks_tests)

class TestOrthoGroup:

    def test_reproducibility(self):
        if False:
            while True:
                i = 10
        seed = 514
        np.random.seed(seed)
        x = ortho_group.rvs(3)
        x2 = ortho_group.rvs(3, random_state=seed)
        assert_almost_equal(np.linalg.det(x), -1)
        expected = np.array([[0.381686, -0.090374, 0.919863], [0.905794, -0.161537, -0.391718], [-0.183993, -0.98272, -0.020204]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_dim(self):
        if False:
            return 10
        assert_raises(ValueError, ortho_group.rvs, None)
        assert_raises(ValueError, ortho_group.rvs, (2, 2))
        assert_raises(ValueError, ortho_group.rvs, 1)
        assert_raises(ValueError, ortho_group.rvs, 2.5)

    def test_frozen_matrix(self):
        if False:
            while True:
                i = 10
        dim = 7
        frozen = ortho_group(dim)
        frozen_seed = ortho_group(dim, seed=1234)
        rvs1 = frozen.rvs(random_state=1234)
        rvs2 = ortho_group.rvs(dim, random_state=1234)
        rvs3 = frozen_seed.rvs(size=1)
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_det_and_ortho(self):
        if False:
            return 10
        xs = [[ortho_group.rvs(dim) for i in range(10)] for dim in range(2, 12)]
        dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
        assert_allclose(np.fabs(dets), np.ones(dets.shape), rtol=1e-13)
        for xx in xs:
            for x in xx:
                assert_array_almost_equal(np.dot(x, x.T), np.eye(x.shape[0]))

    @pytest.mark.parametrize('dim', [2, 5, 10, 20])
    def test_det_distribution_gh18272(self, dim):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(6796248956179332344)
        dist = ortho_group(dim=dim)
        rvs = dist.rvs(size=5000, random_state=rng)
        dets = scipy.linalg.det(rvs)
        k = np.sum(dets > 0)
        n = len(dets)
        res = stats.binomtest(k, n)
        (low, high) = res.proportion_ci(confidence_level=0.95)
        assert low < 0.5 < high

    def test_haar(self):
        if False:
            while True:
                i = 10
        dim = 5
        samples = 1000
        ks_prob = 0.05
        np.random.seed(518)
        xs = ortho_group.rvs(dim, size=samples)
        els = ((0, 0), (0, 2), (1, 4), (2, 3))
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for (er, ec) in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob] * len(pairs), ks_tests)

    @pytest.mark.slow
    def test_pairwise_distances(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(514)

        def random_ortho(dim):
            if False:
                print('Hello World!')
            (u, _s, v) = np.linalg.svd(np.random.normal(size=(dim, dim)))
            return np.dot(u, v)
        for dim in range(2, 6):

            def generate_test_statistics(rvs, N=1000, eps=1e-10):
                if False:
                    for i in range(10):
                        print('nop')
                stats = np.array([np.sum((rvs(dim=dim) - rvs(dim=dim)) ** 2) for _ in range(N)])
                stats += np.random.uniform(-eps, eps, size=stats.shape)
                return stats
            expected = generate_test_statistics(random_ortho)
            actual = generate_test_statistics(scipy.stats.ortho_group.rvs)
            (_D, p) = scipy.stats.ks_2samp(expected, actual)
            assert_array_less(0.05, p)

class TestRandomCorrelation:

    def test_reproducibility(self):
        if False:
            print('Hello World!')
        np.random.seed(514)
        eigs = (0.5, 0.8, 1.2, 1.5)
        x = random_correlation.rvs(eigs)
        x2 = random_correlation.rvs(eigs, random_state=514)
        expected = np.array([[1.0, -0.184851, 0.109017, -0.227494], [-0.184851, 1.0, 0.231236, 0.326669], [0.109017, 0.231236, 1.0, -0.178912], [-0.227494, 0.326669, -0.178912, 1.0]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_eigs(self):
        if False:
            print('Hello World!')
        assert_raises(ValueError, random_correlation.rvs, None)
        assert_raises(ValueError, random_correlation.rvs, 'test')
        assert_raises(ValueError, random_correlation.rvs, 2.5)
        assert_raises(ValueError, random_correlation.rvs, [2.5])
        assert_raises(ValueError, random_correlation.rvs, [[1, 2], [3, 4]])
        assert_raises(ValueError, random_correlation.rvs, [2.5, -0.5])
        assert_raises(ValueError, random_correlation.rvs, [1, 2, 0.1])

    def test_frozen_matrix(self):
        if False:
            return 10
        eigs = (0.5, 0.8, 1.2, 1.5)
        frozen = random_correlation(eigs)
        frozen_seed = random_correlation(eigs, seed=514)
        rvs1 = random_correlation.rvs(eigs, random_state=514)
        rvs2 = frozen.rvs(random_state=514)
        rvs3 = frozen_seed.rvs()
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_definition(self):
        if False:
            i = 10
            return i + 15

        def norm(i, e):
            if False:
                return 10
            return i * e / sum(e)
        np.random.seed(123)
        eigs = [norm(i, np.random.uniform(size=i)) for i in range(2, 6)]
        eigs.append([4, 0, 0, 0])
        ones = [[1.0] * len(e) for e in eigs]
        xs = [random_correlation.rvs(e) for e in eigs]
        dets = [np.fabs(np.linalg.det(x)) for x in xs]
        dets_known = [np.prod(e) for e in eigs]
        assert_allclose(dets, dets_known, rtol=1e-13, atol=1e-13)
        diags = [np.diag(x) for x in xs]
        for (a, b) in zip(diags, ones):
            assert_allclose(a, b, rtol=1e-13)
        for x in xs:
            assert_allclose(x, x.T, rtol=1e-13)

    def test_to_corr(self):
        if False:
            print('Hello World!')
        m = np.array([[0.1, 0], [0, 1]], dtype=float)
        m = random_correlation._to_corr(m)
        assert_allclose(m, np.array([[1, 0], [0, 0.1]]))
        with np.errstate(over='ignore'):
            g = np.array([[0, 1], [-1, 0]])
            m0 = np.array([[1e+300, 0], [0, np.nextafter(1, 0)]], dtype=float)
            m = random_correlation._to_corr(m0.copy())
            assert_allclose(m, g.T.dot(m0).dot(g))
            m0 = np.array([[0.9, 1e+300], [1e+300, 1.1]], dtype=float)
            m = random_correlation._to_corr(m0.copy())
            assert_allclose(m, g.T.dot(m0).dot(g))
        m0 = np.array([[2, 1], [1, 2]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m[0, 0], 1)
        m0 = np.array([[2 + 1e-07, 1], [1, 2]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m[0, 0], 1)

class TestUniformDirection:

    @pytest.mark.parametrize('dim', [1, 3])
    @pytest.mark.parametrize('size', [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        if False:
            i = 10
            return i + 15
        rng = np.random.default_rng(2777937887058094419)
        uniform_direction_dist = uniform_direction(dim, seed=rng)
        samples = uniform_direction_dist.rvs(size)
        (mean, cov) = (np.zeros(dim), np.eye(dim))
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        assert samples.shape == expected_shape
        norms = np.linalg.norm(samples, axis=-1)
        assert_allclose(norms, 1.0)

    @pytest.mark.parametrize('dim', [None, 0, (2, 2), 2.5])
    def test_invalid_dim(self, dim):
        if False:
            while True:
                i = 10
        message = 'Dimension of vector must be specified, and must be an integer greater than 0.'
        with pytest.raises(ValueError, match=message):
            uniform_direction.rvs(dim)

    def test_frozen_distribution(self):
        if False:
            while True:
                i = 10
        dim = 5
        frozen = uniform_direction(dim)
        frozen_seed = uniform_direction(dim, seed=514)
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = uniform_direction.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs()
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    @pytest.mark.parametrize('dim', [2, 5, 8])
    def test_uniform(self, dim):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(1036978481269651776)
        spherical_dist = uniform_direction(dim, seed=rng)
        (v1, v2) = spherical_dist.rvs(size=2)
        v2 -= v1 @ v2 * v1
        v2 /= np.linalg.norm(v2)
        assert_allclose(v1 @ v2, 0, atol=1e-14)
        samples = spherical_dist.rvs(size=10000)
        s1 = samples @ v1
        s2 = samples @ v2
        angles = np.arctan2(s1, s2)
        angles += np.pi
        angles /= 2 * np.pi
        uniform_dist = uniform()
        kstest_result = kstest(angles, uniform_dist.cdf)
        assert kstest_result.pvalue > 0.05

class TestUnitaryGroup:

    def test_reproducibility(self):
        if False:
            while True:
                i = 10
        np.random.seed(514)
        x = unitary_group.rvs(3)
        x2 = unitary_group.rvs(3, random_state=514)
        expected = np.array([[0.308771 + 0.360312j, 0.044021 + 0.622082j, 0.160327 + 0.600173j], [0.732757 + 0.297107j, 0.076692 - 0.4614j, -0.394349 + 0.022613j], [-0.148844 + 0.357037j, -0.284602 - 0.557949j, 0.607051 + 0.299257j]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_dim(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, unitary_group.rvs, None)
        assert_raises(ValueError, unitary_group.rvs, (2, 2))
        assert_raises(ValueError, unitary_group.rvs, 1)
        assert_raises(ValueError, unitary_group.rvs, 2.5)

    def test_frozen_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        dim = 7
        frozen = unitary_group(dim)
        frozen_seed = unitary_group(dim, seed=514)
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = unitary_group.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs(size=1)
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_unitarity(self):
        if False:
            while True:
                i = 10
        xs = [unitary_group.rvs(dim) for dim in range(2, 12) for i in range(3)]
        for x in xs:
            assert_allclose(np.dot(x, x.conj().T), np.eye(x.shape[0]), atol=1e-15)

    def test_haar(self):
        if False:
            print('Hello World!')
        dim = 5
        samples = 1000
        np.random.seed(514)
        xs = unitary_group.rvs(dim, size=samples)
        eigs = np.vstack([scipy.linalg.eigvals(x) for x in xs])
        x = np.arctan2(eigs.imag, eigs.real)
        res = kstest(x.ravel(), uniform(-np.pi, 2 * np.pi).cdf)
        assert_(res.pvalue > 0.05)

class TestMultivariateT:
    PDF_TESTS = [([[1, 2], [4, 1], [2, 1], [2, 4], [1, 4], [4, 1], [3, 2], [3, 3], [4, 4], [5, 1]], [0, 0], [[1, 0], [0, 1]], 4, [0.013972450422333742, 0.001099872190679333, 0.013972450422333742, 0.0007368284402402561, 0.001099872190679333, 0.001099872190679333, 0.0020732579600816823, 0.0009566037150527143, 0.00021831953784896499, 0.0003772561614030115]), ([[0.9718, 0.1298, 0.8134], [0.4922, 0.5522, 0.7185], [0.301, 0.1491, 0.5008], [0.5971, 0.2585, 0.894], [0.5434, 0.5287, 0.9507]], [-1, 1, 50], [[1.0, 0.5, 0.25], [0.5, 1.0, -0.1], [0.25, -0.1, 1.0]], 8, [6.960927969746777e-16, 7.370073905220737e-16, 6.952290996266917e-16, 7.421229355799831e-16, 7.703967515402212e-16])]

    @pytest.mark.parametrize('x, loc, shape, df, ans', PDF_TESTS)
    def test_pdf_correctness(self, x, loc, shape, df, ans):
        if False:
            print('Hello World!')
        dist = multivariate_t(loc, shape, df, seed=0)
        val = dist.pdf(x)
        assert_array_almost_equal(val, ans)

    @pytest.mark.parametrize('x, loc, shape, df, ans', PDF_TESTS)
    def test_logpdf_correct(self, x, loc, shape, df, ans):
        if False:
            print('Hello World!')
        dist = multivariate_t(loc, shape, df, seed=0)
        val1 = dist.pdf(x)
        val2 = dist.logpdf(x)
        assert_array_almost_equal(np.log(val1), val2)

    def test_mvt_with_df_one_is_cauchy(self):
        if False:
            for i in range(10):
                print('nop')
        x = [9, 7, 4, 1, -3, 9, 0, -3, -1, 3]
        val = multivariate_t.pdf(x, df=1)
        ans = cauchy.pdf(x)
        assert_array_almost_equal(val, ans)

    def test_mvt_with_high_df_is_approx_normal(self):
        if False:
            while True:
                i = 10
        P_VAL_MIN = 0.1
        dist = multivariate_t(0, 1, df=100000, seed=1)
        samples = dist.rvs(size=100000)
        (_, p) = normaltest(samples)
        assert p > P_VAL_MIN
        dist = multivariate_t([-2, 3], [[10, -1], [-1, 10]], df=100000, seed=42)
        samples = dist.rvs(size=100000)
        (_, p) = normaltest(samples)
        assert (p > P_VAL_MIN).all()

    @patch('scipy.stats.multivariate_normal._logpdf')
    def test_mvt_with_inf_df_calls_normal(self, mock):
        if False:
            for i in range(10):
                print('nop')
        dist = multivariate_t(0, 1, df=np.inf, seed=7)
        assert isinstance(dist, multivariate_normal_frozen)
        multivariate_t.pdf(0, df=np.inf)
        assert mock.call_count == 1
        multivariate_t.logpdf(0, df=np.inf)
        assert mock.call_count == 2

    def test_shape_correctness(self):
        if False:
            while True:
                i = 10
        dim = 4
        loc = np.zeros(dim)
        shape = np.eye(dim)
        df = 4.5
        x = np.zeros(dim)
        res = multivariate_t(loc, shape, df).pdf(x)
        assert np.isscalar(res)
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert np.isscalar(res)
        n_samples = 7
        x = np.random.random((n_samples, dim))
        res = multivariate_t(loc, shape, df).pdf(x)
        assert res.shape == (n_samples,)
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert res.shape == (n_samples,)
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs()
        assert np.isscalar(res)
        size = 7
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs(size=size)
        assert res.shape == (size,)

    def test_default_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        dist = multivariate_t()
        assert_equal(dist.loc, [0])
        assert_equal(dist.shape, [[1]])
        assert dist.df == 1
    DEFAULT_ARGS_TESTS = [(None, None, None, 0, 1, 1), (None, None, 7, 0, 1, 7), (None, [[7, 0], [0, 7]], None, [0, 0], [[7, 0], [0, 7]], 1), (None, [[7, 0], [0, 7]], 7, [0, 0], [[7, 0], [0, 7]], 7), ([7, 7], None, None, [7, 7], [[1, 0], [0, 1]], 1), ([7, 7], None, 7, [7, 7], [[1, 0], [0, 1]], 7), ([7, 7], [[7, 0], [0, 7]], None, [7, 7], [[7, 0], [0, 7]], 1), ([7, 7], [[7, 0], [0, 7]], 7, [7, 7], [[7, 0], [0, 7]], 7)]

    @pytest.mark.parametrize('loc, shape, df, loc_ans, shape_ans, df_ans', DEFAULT_ARGS_TESTS)
    def test_default_args(self, loc, shape, df, loc_ans, shape_ans, df_ans):
        if False:
            i = 10
            return i + 15
        dist = multivariate_t(loc=loc, shape=shape, df=df)
        assert_equal(dist.loc, loc_ans)
        assert_equal(dist.shape, shape_ans)
        assert dist.df == df_ans
    ARGS_SHAPES_TESTS = [(-1, 2, 3, [-1], [[2]], 3), ([-1], [2], 3, [-1], [[2]], 3), (np.array([-1]), np.array([2]), 3, [-1], [[2]], 3)]

    @pytest.mark.parametrize('loc, shape, df, loc_ans, shape_ans, df_ans', ARGS_SHAPES_TESTS)
    def test_scalar_list_and_ndarray_arguments(self, loc, shape, df, loc_ans, shape_ans, df_ans):
        if False:
            return 10
        dist = multivariate_t(loc, shape, df)
        assert_equal(dist.loc, loc_ans)
        assert_equal(dist.shape, shape_ans)
        assert_equal(dist.df, df_ans)

    def test_argument_error_handling(self):
        if False:
            for i in range(10):
                print('nop')
        loc = [[1, 1]]
        assert_raises(ValueError, multivariate_t, **dict(loc=loc))
        shape = [[1, 1], [2, 2], [3, 3]]
        assert_raises(ValueError, multivariate_t, **dict(loc=loc, shape=shape))
        loc = np.zeros(2)
        shape = np.eye(2)
        df = -1
        assert_raises(ValueError, multivariate_t, **dict(loc=loc, shape=shape, df=df))
        df = 0
        assert_raises(ValueError, multivariate_t, **dict(loc=loc, shape=shape, df=df))

    def test_reproducibility(self):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.RandomState(4)
        loc = rng.uniform(size=3)
        shape = np.eye(3)
        dist1 = multivariate_t(loc, shape, df=3, seed=2)
        dist2 = multivariate_t(loc, shape, df=3, seed=2)
        samples1 = dist1.rvs(size=10)
        samples2 = dist2.rvs(size=10)
        assert_equal(samples1, samples2)

    def test_allow_singular(self):
        if False:
            for i in range(10):
                print('nop')
        args = dict(loc=[0, 0], shape=[[0, 0], [0, 1]], df=1, allow_singular=False)
        assert_raises(np.linalg.LinAlgError, multivariate_t, **args)

    @pytest.mark.parametrize('size', [(10, 3), (5, 6, 4, 3)])
    @pytest.mark.parametrize('dim', [2, 3, 4, 5])
    @pytest.mark.parametrize('df', [1.0, 2.0, np.inf])
    def test_rvs(self, size, dim, df):
        if False:
            i = 10
            return i + 15
        dist = multivariate_t(np.zeros(dim), np.eye(dim), df)
        rvs = dist.rvs(size=size)
        assert rvs.shape == size + (dim,)

    def test_cdf_signs(self):
        if False:
            i = 10
            return i + 15
        mean = np.zeros(3)
        cov = np.eye(3)
        df = 10
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        expected_signs = np.array([1, -1, -1, 1])
        cdf = multivariate_normal.cdf(b, mean, cov, df, lower_limit=a)
        assert_allclose(cdf, cdf[0] * expected_signs)

    @pytest.mark.parametrize('dim', [1, 2, 5, 10])
    def test_cdf_against_multivariate_normal(self, dim):
        if False:
            return 10
        self.cdf_against_mvn_test(dim)

    @pytest.mark.parametrize('dim', [3, 6, 9])
    def test_cdf_against_multivariate_normal_singular(self, dim):
        if False:
            i = 10
            return i + 15
        self.cdf_against_mvn_test(3, True)

    def cdf_against_mvn_test(self, dim, singular=False):
        if False:
            return 10
        rng = np.random.default_rng(413722918996573)
        n = 3
        w = 10 ** rng.uniform(-2, 1, size=dim)
        cov = _random_covariance(dim, w, rng, singular)
        mean = 10 ** rng.uniform(-1, 2, size=dim) * np.sign(rng.normal(size=dim))
        a = -10 ** rng.uniform(-1, 2, size=(n, dim)) + mean
        b = 10 ** rng.uniform(-1, 2, size=(n, dim)) + mean
        res = stats.multivariate_t.cdf(b, mean, cov, df=10000, lower_limit=a, allow_singular=True, random_state=rng)
        ref = stats.multivariate_normal.cdf(b, mean, cov, allow_singular=True, lower_limit=a)
        assert_allclose(res, ref, atol=0.0005)

    def test_cdf_against_univariate_t(self):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(413722918996573)
        cov = 2
        mean = 0
        x = rng.normal(size=10, scale=np.sqrt(cov))
        df = 3
        res = stats.multivariate_t.cdf(x, mean, cov, df, lower_limit=-np.inf, random_state=rng)
        ref = stats.t.cdf(x, df, mean, np.sqrt(cov))
        incorrect = stats.norm.cdf(x, mean, np.sqrt(cov))
        assert_allclose(res, ref, atol=0.0005)
        assert np.all(np.abs(res - incorrect) > 0.001)

    @pytest.mark.parametrize('dim', [2, 3, 5, 10])
    @pytest.mark.parametrize('seed', [3363958638, 7891119608, 3887698049, 5013150848, 1495033423, 6170824608])
    @pytest.mark.parametrize('singular', [False, True])
    def test_cdf_against_qsimvtv(self, dim, seed, singular):
        if False:
            while True:
                i = 10
        if singular and seed != 3363958638:
            pytest.skip('Agreement with qsimvtv is not great in singular case')
        rng = np.random.default_rng(seed)
        w = 10 ** rng.uniform(-2, 2, size=dim)
        cov = _random_covariance(dim, w, rng, singular)
        mean = rng.random(dim)
        a = -rng.random(dim)
        b = rng.random(dim)
        df = rng.random() * 5
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng, allow_singular=True)
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, np.inf * a, b - mean, rng)[0]
        assert_allclose(res, ref, atol=0.0002, rtol=0.001)
        res = stats.multivariate_t.cdf(b, mean, cov, df, lower_limit=a, random_state=rng, allow_singular=True)
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, a - mean, b - mean, rng)[0]
        assert_allclose(res, ref, atol=0.0001, rtol=0.001)

    def test_cdf_against_generic_integrators(self):
        if False:
            i = 10
            return i + 15
        dim = 3
        rng = np.random.default_rng(41372291899657)
        w = 10 ** rng.uniform(-1, 1, size=dim)
        cov = _random_covariance(dim, w, rng, singular=True)
        mean = rng.random(dim)
        a = -rng.random(dim)
        b = rng.random(dim)
        df = rng.random() * 5
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng, lower_limit=a)

        def integrand(x):
            if False:
                while True:
                    i = 10
            return stats.multivariate_t.pdf(x.T, mean, cov, df)
        ref = qmc_quad(integrand, a, b, qrng=stats.qmc.Halton(d=dim, seed=rng))
        assert_allclose(res, ref.integral, rtol=0.001)

        def integrand(*zyx):
            if False:
                return 10
            return stats.multivariate_t.pdf(zyx[::-1], mean, cov, df)
        ref = tplquad(integrand, a[0], b[0], a[1], b[1], a[2], b[2])
        assert_allclose(res, ref[0], rtol=0.001)

    def test_against_matlab(self):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(2967390923)
        cov = np.array([[6.21786909, 0.2333667, 7.95506077], [0.2333667, 29.67390923, 16.53946426], [7.95506077, 16.53946426, 19.17725252]])
        df = 1.9559939787727658
        dist = stats.multivariate_t(shape=cov, df=df)
        res = dist.cdf([0, 0, 0], random_state=rng)
        ref = 0.2523
        assert_allclose(res, ref, rtol=0.001)

    def test_frozen(self):
        if False:
            for i in range(10):
                print('nop')
        seed = 4137229573
        rng = np.random.default_rng(seed)
        loc = rng.uniform(size=3)
        x = rng.uniform(size=3) + loc
        shape = np.eye(3)
        df = rng.random()
        args = (loc, shape, df)
        rng_frozen = np.random.default_rng(seed)
        rng_unfrozen = np.random.default_rng(seed)
        dist = stats.multivariate_t(*args, seed=rng_frozen)
        assert_equal(dist.cdf(x), multivariate_t.cdf(x, *args, random_state=rng_unfrozen))

    def test_vectorized(self):
        if False:
            for i in range(10):
                print('nop')
        dim = 4
        n = (2, 3)
        rng = np.random.default_rng(413722918996573)
        A = rng.random(size=(dim, dim))
        cov = A @ A.T
        mean = rng.random(dim)
        x = rng.random(n + (dim,))
        df = rng.random() * 5
        res = stats.multivariate_t.cdf(x, mean, cov, df, random_state=rng)

        def _cdf_1d(x):
            if False:
                for i in range(10):
                    print('nop')
            return _qsimvtv(10000, df, cov, -np.inf * x, x - mean, rng)[0]
        ref = np.apply_along_axis(_cdf_1d, -1, x)
        assert_allclose(res, ref, atol=0.0001, rtol=0.001)

    @pytest.mark.parametrize('dim', (3, 7))
    def test_against_analytical(self, dim):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(413722918996573)
        A = scipy.linalg.toeplitz(c=[1] + [0.5] * (dim - 1))
        res = stats.multivariate_t(shape=A).cdf([0] * dim, random_state=rng)
        ref = 1 / (dim + 1)
        assert_allclose(res, ref, rtol=5e-05)

    def test_entropy_inf_df(self):
        if False:
            for i in range(10):
                print('nop')
        cov = np.eye(3, 3)
        df = np.inf
        mvt_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mvn_entropy = stats.multivariate_normal.entropy(None, cov)
        assert mvt_entropy == mvn_entropy

    @pytest.mark.parametrize('df', [1, 10, 100])
    def test_entropy_1d(self, df):
        if False:
            return 10
        mvt_entropy = stats.multivariate_t.entropy(shape=1.0, df=df)
        t_entropy = stats.t.entropy(df=df)
        assert_allclose(mvt_entropy, t_entropy, rtol=1e-13)

    @pytest.mark.parametrize('df, cov, ref, tol', [(10, np.eye(2, 2), 3.0378770664093313, 1e-14), (100, np.array([[0.5, 1], [1, 10]]), 3.55102424550609, 1e-08)])
    def test_entropy_vs_numerical_integration(self, df, cov, ref, tol):
        if False:
            i = 10
            return i + 15
        loc = np.zeros((2,))
        mvt = stats.multivariate_t(loc, cov, df)
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    @pytest.mark.parametrize('df, dim, ref, tol', [(10, 1, 1.5212624929756808, 1e-15), (100, 1, 1.4289633653182439, 1e-13), (500, 1, 1.420939531869349, 1e-14), (1e+20, 1, 1.4189385332046727, 1e-15), (1e+100, 1, 1.4189385332046727, 1e-15), (10, 10, 15.069150450832911, 1e-15), (1000, 10, 14.19936546446673, 1e-13), (1e+20, 10, 14.189385332046728, 1e-15), (1e+100, 10, 14.189385332046728, 1e-15), (10, 100, 148.28902883192654, 1e-15), (1000, 100, 141.99155538003762, 1e-14), (1e+20, 100, 141.8938533204673, 1e-15), (1e+100, 100, 141.8938533204673, 1e-15)])
    def test_extreme_entropy(self, df, dim, ref, tol):
        if False:
            return 10
        mvt = stats.multivariate_t(shape=np.eye(dim), df=df)
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    def test_entropy_with_covariance(self):
        if False:
            return 10
        _A = np.array([[1.42, 0.09, -0.49, 0.17, 0.74], [-1.13, -0.01, 0.71, 0.4, -0.56], [1.07, 0.44, -0.28, -0.44, 0.29], [-1.5, -0.94, -0.67, 0.73, -1.1], [0.17, -0.08, 1.46, -0.32, 1.36]])
        cov = _A @ _A.T
        df = 1e+20
        mul_t_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mul_norm_entropy = multivariate_normal(None, cov=cov).entropy()
        assert_allclose(mul_t_entropy, mul_norm_entropy, rtol=1e-15)
        df1 = 765
        df2 = 768
        _entropy1 = stats.multivariate_t.entropy(shape=cov, df=df1)
        _entropy2 = stats.multivariate_t.entropy(shape=cov, df=df2)
        assert_allclose(_entropy1, _entropy2, rtol=1e-05)

class TestMultivariateHypergeom:

    @pytest.mark.parametrize('x, m, n, expected', [([3, 4], [5, 10], 7, -1.119814), ([3, 4], [5, 10], 0, -np.inf), ([-3, 4], [5, 10], 7, -np.inf), ([3, 4], [-5, 10], 7, np.nan), ([[1, 2], [3, 4]], [[-4, -6], [-5, -10]], [3, 7], [np.nan, np.nan]), ([-3, 4], [-5, 10], 1, np.nan), ([1, 11], [10, 1], 12, np.nan), ([1, 11], [10, -1], 12, np.nan), ([3, 4], [5, 10], -7, np.nan), ([3, 3], [5, 10], 7, -np.inf)])
    def test_logpmf(self, x, m, n, expected):
        if False:
            return 10
        vals = multivariate_hypergeom.logpmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-06)

    def test_reduces_hypergeom(self):
        if False:
            print('Hello World!')
        val1 = multivariate_hypergeom.pmf(x=[3, 1], m=[10, 5], n=4)
        val2 = hypergeom.pmf(k=3, M=15, n=4, N=10)
        assert_allclose(val1, val2, rtol=1e-08)
        val1 = multivariate_hypergeom.pmf(x=[7, 3], m=[15, 10], n=10)
        val2 = hypergeom.pmf(k=7, M=25, n=10, N=15)
        assert_allclose(val1, val2, rtol=1e-08)

    def test_rvs(self):
        if False:
            for i in range(10):
                print('nop')
        rv = multivariate_hypergeom(m=[3, 5], n=4)
        rvs = rv.rvs(size=1000, random_state=123)
        assert_allclose(rvs.mean(0), rv.mean(), rtol=0.01)

    def test_rvs_broadcasting(self):
        if False:
            i = 10
            return i + 15
        rv = multivariate_hypergeom(m=[[3, 5], [5, 10]], n=[4, 9])
        rvs = rv.rvs(size=(1000, 2), random_state=123)
        assert_allclose(rvs.mean(0), rv.mean(), rtol=0.01)

    @pytest.mark.parametrize('m, n', (([0, 0, 20, 0, 0], 5), ([0, 0, 0, 0, 0], 0), ([0, 0], 0), ([0], 0)))
    def test_rvs_gh16171(self, m, n):
        if False:
            i = 10
            return i + 15
        res = multivariate_hypergeom.rvs(m, n)
        m = np.asarray(m)
        res_ex = m.copy()
        res_ex[m != 0] = n
        assert_equal(res, res_ex)

    @pytest.mark.parametrize('x, m, n, expected', [([5], [5], 5, 1), ([3, 4], [5, 10], 7, 0.3263403), ([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]], [5, 10], [[8, 8], [8, 2]], [[0.3916084, 0.006993007], [0, 0.4761905]]), (np.array([], dtype=int), np.array([], dtype=int), 0, []), ([1, 2], [4, 5], 5, 0), ([3, 3, 0], [5, 6, 7], 6, 0.01077354)])
    def test_pmf(self, x, m, n, expected):
        if False:
            return 10
        vals = multivariate_hypergeom.pmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-07)

    @pytest.mark.parametrize('x, m, n, expected', [([3, 4], [[5, 10], [10, 15]], 7, [0.3263403, 0.3407531]), ([[1], [2]], [[3], [4]], [1, 3], [1.0, 0.0]), ([[[1], [2]]], [[3], [4]], [1, 3], [[1.0, 0.0]]), ([[1], [2]], [[[[3]]]], [1, 3], [[[1.0, 0.0]]])])
    def test_pmf_broadcasting(self, x, m, n, expected):
        if False:
            for i in range(10):
                print('nop')
        vals = multivariate_hypergeom.pmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-07)

    def test_cov(self):
        if False:
            print('Hello World!')
        cov1 = multivariate_hypergeom.cov(m=[3, 7, 10], n=12)
        cov2 = [[0.64421053, -0.26526316, -0.37894737], [-0.26526316, 1.14947368, -0.88421053], [-0.37894737, -0.88421053, 1.26315789]]
        assert_allclose(cov1, cov2, rtol=1e-08)

    def test_cov_broadcasting(self):
        if False:
            for i in range(10):
                print('nop')
        cov1 = multivariate_hypergeom.cov(m=[[7, 9], [10, 15]], n=[8, 12])
        cov2 = [[[1.05, -1.05], [-1.05, 1.05]], [[1.56, -1.56], [-1.56, 1.56]]]
        assert_allclose(cov1, cov2, rtol=1e-08)
        cov3 = multivariate_hypergeom.cov(m=[[4], [5]], n=[4, 5])
        cov4 = [[[0.0]], [[0.0]]]
        assert_allclose(cov3, cov4, rtol=1e-08)
        cov5 = multivariate_hypergeom.cov(m=[7, 9], n=[8, 12])
        cov6 = [[[1.05, -1.05], [-1.05, 1.05]], [[0.7875, -0.7875], [-0.7875, 0.7875]]]
        assert_allclose(cov5, cov6, rtol=1e-08)

    def test_var(self):
        if False:
            for i in range(10):
                print('nop')
        var0 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var1 = hypergeom.var(M=15, n=4, N=10)
        assert_allclose(var0, var1, rtol=1e-08)

    def test_var_broadcasting(self):
        if False:
            i = 10
            return i + 15
        var0 = multivariate_hypergeom.var(m=[10, 5], n=[4, 8])
        var1 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var2 = multivariate_hypergeom.var(m=[10, 5], n=8)
        assert_allclose(var0[0], var1, rtol=1e-08)
        assert_allclose(var0[1], var2, rtol=1e-08)
        var3 = multivariate_hypergeom.var(m=[[10, 5], [10, 14]], n=[4, 8])
        var4 = [[0.6984127, 0.6984127], [1.352657, 1.352657]]
        assert_allclose(var3, var4, rtol=1e-08)
        var5 = multivariate_hypergeom.var(m=[[5], [10]], n=[5, 10])
        var6 = [[0.0], [0.0]]
        assert_allclose(var5, var6, rtol=1e-08)

    def test_mean(self):
        if False:
            return 10
        mean0 = multivariate_hypergeom.mean(m=[10, 5], n=4)
        mean1 = hypergeom.mean(M=15, n=4, N=10)
        assert_allclose(mean0[0], mean1, rtol=1e-08)
        mean2 = multivariate_hypergeom.mean(m=[12, 8], n=10)
        mean3 = [12.0 * 10.0 / 20.0, 8.0 * 10.0 / 20.0]
        assert_allclose(mean2, mean3, rtol=1e-08)

    def test_mean_broadcasting(self):
        if False:
            while True:
                i = 10
        mean0 = multivariate_hypergeom.mean(m=[[3, 5], [10, 5]], n=[4, 8])
        mean1 = [[3.0 * 4.0 / 8.0, 5.0 * 4.0 / 8.0], [10.0 * 8.0 / 15.0, 5.0 * 8.0 / 15.0]]
        assert_allclose(mean0, mean1, rtol=1e-08)

    def test_mean_edge_cases(self):
        if False:
            print('Hello World!')
        mean0 = multivariate_hypergeom.mean(m=[0, 0, 0], n=0)
        assert_equal(mean0, [0.0, 0.0, 0.0])
        mean1 = multivariate_hypergeom.mean(m=[1, 0, 0], n=2)
        assert_equal(mean1, [np.nan, np.nan, np.nan])
        mean2 = multivariate_hypergeom.mean(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(mean2, [[np.nan, np.nan, np.nan], [1.0, 0.0, 1.0]], rtol=1e-17)
        mean3 = multivariate_hypergeom.mean(m=np.array([], dtype=int), n=0)
        assert_equal(mean3, [])
        assert_(mean3.shape == (0,))

    def test_var_edge_cases(self):
        if False:
            while True:
                i = 10
        var0 = multivariate_hypergeom.var(m=[0, 0, 0], n=0)
        assert_allclose(var0, [0.0, 0.0, 0.0], rtol=1e-16)
        var1 = multivariate_hypergeom.var(m=[1, 0, 0], n=2)
        assert_equal(var1, [np.nan, np.nan, np.nan])
        var2 = multivariate_hypergeom.var(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(var2, [[np.nan, np.nan, np.nan], [0.0, 0.0, 0.0]], rtol=1e-17)
        var3 = multivariate_hypergeom.var(m=np.array([], dtype=int), n=0)
        assert_equal(var3, [])
        assert_(var3.shape == (0,))

    def test_cov_edge_cases(self):
        if False:
            print('Hello World!')
        cov0 = multivariate_hypergeom.cov(m=[1, 0, 0], n=1)
        cov1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert_allclose(cov0, cov1, rtol=1e-17)
        cov3 = multivariate_hypergeom.cov(m=[0, 0, 0], n=0)
        cov4 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert_equal(cov3, cov4)
        cov5 = multivariate_hypergeom.cov(m=np.array([], dtype=int), n=0)
        cov6 = np.array([], dtype=np.float64).reshape(0, 0)
        assert_allclose(cov5, cov6, rtol=1e-17)
        assert_(cov5.shape == (0, 0))

    def test_frozen(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        n = 12
        m = [7, 9, 11, 13]
        x = [[0, 0, 0, 12], [0, 0, 1, 11], [0, 1, 1, 10], [1, 1, 1, 9], [1, 1, 2, 8]]
        x = np.asarray(x, dtype=int)
        mhg_frozen = multivariate_hypergeom(m, n)
        assert_allclose(mhg_frozen.pmf(x), multivariate_hypergeom.pmf(x, m, n))
        assert_allclose(mhg_frozen.logpmf(x), multivariate_hypergeom.logpmf(x, m, n))
        assert_allclose(mhg_frozen.var(), multivariate_hypergeom.var(m, n))
        assert_allclose(mhg_frozen.cov(), multivariate_hypergeom.cov(m, n))

    def test_invalid_params(self):
        if False:
            while True:
                i = 10
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, 10, 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, [10], 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, [5, 4], [10], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5.5, 4.5], [10, 15], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4], [10.5, 15.5], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4], [10, 15], 5.5)

class TestRandomTable:

    def get_rng(self):
        if False:
            i = 10
            return i + 15
        return np.random.default_rng(628174795866951638)

    def test_process_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        message = '`row` must be one-dimensional'
        with pytest.raises(ValueError, match=message):
            random_table([[1, 2]], [1, 2])
        message = '`col` must be one-dimensional'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [[1, 2]])
        message = 'each element of `row` must be non-negative'
        with pytest.raises(ValueError, match=message):
            random_table([1, -1], [1, 2])
        message = 'each element of `col` must be non-negative'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1, -2])
        message = 'sums over `row` and `col` must be equal'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1, 0])
        message = 'each element of `row` must be an integer'
        with pytest.raises(ValueError, match=message):
            random_table([2.1, 2.1], [1, 1, 2])
        message = 'each element of `col` must be an integer'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1.1, 1.1, 1])
        row = [1, 3]
        col = [2, 1, 1]
        (r, c, n) = random_table._process_parameters([1, 3], [2, 1, 1])
        assert_equal(row, r)
        assert_equal(col, c)
        assert n == np.sum(row)

    @pytest.mark.parametrize('scale,method', ((1, 'boyett'), (100, 'patefield')))
    def test_process_rvs_method_on_None(self, scale, method):
        if False:
            while True:
                i = 10
        row = np.array([1, 3]) * scale
        col = np.array([2, 1, 1]) * scale
        ct = random_table
        expected = ct.rvs(row, col, method=method, random_state=1)
        got = ct.rvs(row, col, method=None, random_state=1)
        assert_equal(expected, got)

    def test_process_rvs_method_bad_argument(self):
        if False:
            print('Hello World!')
        row = [1, 3]
        col = [2, 1, 1]
        message = "'foo' not recognized, must be one of"
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, method='foo')

    @pytest.mark.parametrize('frozen', (True, False))
    @pytest.mark.parametrize('log', (True, False))
    def test_pmf_logpmf(self, frozen, log):
        if False:
            return 10
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs = random_table.rvs(row, col, size=1000, method='boyett', random_state=rng)
        obj = random_table(row, col) if frozen else random_table
        method = getattr(obj, 'logpmf' if log else 'pmf')
        if not frozen:
            original_method = method

            def method(x):
                if False:
                    print('Hello World!')
                return original_method(x, row, col)
        pmf = (lambda x: np.exp(method(x))) if log else method
        (unique_rvs, counts) = np.unique(rvs, axis=0, return_counts=True)
        p = pmf(unique_rvs)
        assert_allclose(p * len(rvs), counts, rtol=0.1)
        p2 = pmf(list(unique_rvs[0]))
        assert_equal(p2, p[0])
        rvs_nd = rvs.reshape((10, 100) + rvs.shape[1:])
        p = pmf(rvs_nd)
        assert p.shape == (10, 100)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                pij = p[i, j]
                rvij = rvs_nd[i, j]
                qij = pmf(rvij)
                assert_equal(pij, qij)
        x = [[0, 1, 1], [2, 1, 3]]
        assert_equal(np.sum(x, axis=-1), row)
        p = pmf(x)
        assert p == 0
        x = [[0, 1, 2], [1, 2, 2]]
        assert_equal(np.sum(x, axis=-2), col)
        p = pmf(x)
        assert p == 0
        message = '`x` must be at least two-dimensional'
        with pytest.raises(ValueError, match=message):
            pmf([1])
        message = '`x` must contain only integral values'
        with pytest.raises(ValueError, match=message):
            pmf([[1.1]])
        message = '`x` must contain only integral values'
        with pytest.raises(ValueError, match=message):
            pmf([[np.nan]])
        message = '`x` must contain only non-negative values'
        with pytest.raises(ValueError, match=message):
            pmf([[-1]])
        message = 'shape of `x` must agree with `row`'
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2, 3]])
        message = 'shape of `x` must agree with `col`'
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2], [3, 4]])

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_mean(self, method):
        if False:
            print('Hello World!')
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs = random_table.rvs(row, col, size=1000, method=method, random_state=rng)
        mean = random_table.mean(row, col)
        assert_equal(np.sum(mean), np.sum(row))
        assert_allclose(rvs.mean(0), mean, atol=0.05)
        assert_equal(rvs.sum(axis=-1), np.broadcast_to(row, (1000, 2)))
        assert_equal(rvs.sum(axis=-2), np.broadcast_to(col, (1000, 3)))

    def test_rvs_cov(self):
        if False:
            return 10
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs1 = random_table.rvs(row, col, size=10000, method='boyett', random_state=rng)
        rvs2 = random_table.rvs(row, col, size=10000, method='patefield', random_state=rng)
        cov1 = np.var(rvs1, axis=0)
        cov2 = np.var(rvs2, axis=0)
        assert_allclose(cov1, cov2, atol=0.02)

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_size(self, method):
        if False:
            for i in range(10):
                print('nop')
        row = [2, 6]
        col = [1, 3, 4]
        rv = random_table.rvs(row, col, method=method, random_state=self.get_rng())
        assert rv.shape == (2, 3)
        rv2 = random_table.rvs(row, col, size=1, method=method, random_state=self.get_rng())
        assert rv2.shape == (1, 2, 3)
        assert_equal(rv, rv2[0])
        rv3 = random_table.rvs(row, col, size=0, method=method, random_state=self.get_rng())
        assert rv3.shape == (0, 2, 3)
        rv4 = random_table.rvs(row, col, size=20, method=method, random_state=self.get_rng())
        assert rv4.shape == (20, 2, 3)
        rv5 = random_table.rvs(row, col, size=(4, 5), method=method, random_state=self.get_rng())
        assert rv5.shape == (4, 5, 2, 3)
        assert_allclose(rv5.reshape(20, 2, 3), rv4, rtol=1e-15)
        message = '`size` must be a non-negative integer or `None`'
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=-1, method=method, random_state=self.get_rng())
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=np.nan, method=method, random_state=self.get_rng())

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_method(self, method):
        if False:
            print('Hello World!')
        row = [2, 6]
        col = [1, 3, 4]
        ct = random_table
        rvs = ct.rvs(row, col, size=100000, method=method, random_state=self.get_rng())
        (unique_rvs, counts) = np.unique(rvs, axis=0, return_counts=True)
        p = ct.pmf(unique_rvs, row, col)
        assert_allclose(p * len(rvs), counts, rtol=0.02)

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_with_zeros_in_col_row(self, method):
        if False:
            for i in range(10):
                print('nop')
        row = [0, 1, 0]
        col = [1, 0, 0, 0]
        d = random_table(row, col)
        rv = d.rvs(1000, method=method, random_state=self.get_rng())
        expected = np.zeros((1000, len(row), len(col)))
        expected[...] = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        assert_equal(rv, expected)

    @pytest.mark.parametrize('method', (None, 'boyett', 'patefield'))
    @pytest.mark.parametrize('col', ([], [0]))
    @pytest.mark.parametrize('row', ([], [0]))
    def test_rvs_with_edge_cases(self, method, row, col):
        if False:
            while True:
                i = 10
        d = random_table(row, col)
        rv = d.rvs(10, method=method, random_state=self.get_rng())
        expected = np.zeros((10, len(row), len(col)))
        assert_equal(rv, expected)

    @pytest.mark.parametrize('v', (1, 2))
    def test_rvs_rcont(self, v):
        if False:
            while True:
                i = 10
        import scipy.stats._rcont as _rcont
        row = np.array([1, 3], dtype=np.int64)
        col = np.array([2, 1, 1], dtype=np.int64)
        rvs = getattr(_rcont, f'rvs_rcont{v}')
        ntot = np.sum(row)
        result = rvs(row, col, ntot, 1, self.get_rng())
        assert result.shape == (1, len(row), len(col))
        assert np.sum(result) == ntot

    def test_frozen(self):
        if False:
            return 10
        row = [2, 6]
        col = [1, 3, 4]
        d = random_table(row, col, seed=self.get_rng())
        sample = d.rvs()
        expected = random_table.mean(row, col)
        assert_equal(expected, d.mean())
        expected = random_table.pmf(sample, row, col)
        assert_equal(expected, d.pmf(sample))
        expected = random_table.logpmf(sample, row, col)
        assert_equal(expected, d.logpmf(sample))

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_frozen(self, method):
        if False:
            for i in range(10):
                print('nop')
        row = [2, 6]
        col = [1, 3, 4]
        d = random_table(row, col, seed=self.get_rng())
        expected = random_table.rvs(row, col, size=10, method=method, random_state=self.get_rng())
        got = d.rvs(size=10, method=method)
        assert_equal(expected, got)

def check_pickling(distfn, args):
    if False:
        i = 10
        return i + 15
    rndm = distfn.random_state
    distfn.random_state = 1234
    distfn.rvs(*args, size=8)
    s = pickle.dumps(distfn)
    r0 = distfn.rvs(*args, size=8)
    unpickled = pickle.loads(s)
    r1 = unpickled.rvs(*args, size=8)
    assert_equal(r0, r1)
    distfn.random_state = rndm

def test_random_state_property():
    if False:
        i = 10
        return i + 15
    scale = np.eye(3)
    scale[0, 1] = 0.5
    scale[1, 0] = 0.5
    dists = [[multivariate_normal, ()], [dirichlet, (np.array([1.0]),)], [wishart, (10, scale)], [invwishart, (10, scale)], [multinomial, (5, [0.5, 0.4, 0.1])], [ortho_group, (2,)], [special_ortho_group, (2,)]]
    for (distfn, args) in dists:
        check_random_state_property(distfn, args)
        check_pickling(distfn, args)

class TestVonMises_Fisher:

    @pytest.mark.parametrize('dim', [2, 3, 4, 6])
    @pytest.mark.parametrize('size', [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        if False:
            return 10
        rng = np.random.default_rng(2777937887058094419)
        mu = np.full((dim,), 1 / np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, 1, seed=rng)
        samples = vmf_dist.rvs(size)
        (mean, cov) = (np.zeros(dim), np.eye(dim))
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        assert samples.shape == expected_shape
        norms = np.linalg.norm(samples, axis=-1)
        assert_allclose(norms, 1.0)

    @pytest.mark.parametrize('dim', [5, 8])
    @pytest.mark.parametrize('kappa', [1000000000000000.0, 1e+20, 1e+30])
    def test_sampling_high_concentration(self, dim, kappa):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(2777937887058094419)
        mu = np.full((dim,), 1 / np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, kappa, seed=rng)
        vmf_dist.rvs(10)

    def test_two_dimensional_mu(self):
        if False:
            print('Hello World!')
        mu = np.ones((2, 2))
        msg = "'mu' must have one-dimensional shape."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    def test_wrong_norm_mu(self):
        if False:
            print('Hello World!')
        mu = np.ones((2,))
        msg = "'mu' must be a unit vector of norm 1."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    def test_one_entry_mu(self):
        if False:
            while True:
                i = 10
        mu = np.ones((1,))
        msg = "'mu' must have at least two entries."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    @pytest.mark.parametrize('kappa', [-1, (5, 3)])
    def test_kappa_validation(self, kappa):
        if False:
            print('Hello World!')
        msg = "'kappa' must be a positive scalar."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher([1, 0], kappa)

    @pytest.mark.parametrize('kappa', [0, 0.0])
    def test_kappa_zero(self, kappa):
        if False:
            return 10
        msg = "For 'kappa=0' the von Mises-Fisher distribution becomes the uniform distribution on the sphere surface. Consider using 'scipy.stats.uniform_direction' instead."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher([1, 0], kappa)

    @pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
    def test_invalid_shapes_pdf_logpdf(self, method):
        if False:
            print('Hello World!')
        x = np.array([1.0, 0.0, 0])
        msg = "The dimensionality of the last axis of 'x' must match the dimensionality of the von Mises Fisher distribution."
        with pytest.raises(ValueError, match=msg):
            method(x, [1, 0], 1)

    @pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
    def test_unnormalized_input(self, method):
        if False:
            return 10
        x = np.array([0.5, 0.0])
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        with pytest.raises(ValueError, match=msg):
            method(x, [1, 0], 1)

    @pytest.mark.parametrize('x, mu, kappa, reference', [(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0001, 0.0795854295583605), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 0.0001, 0.07957747141331854), (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 15.915494309189533), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 100, 5.920684802611232e-43), (np.array([1.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0]), 2000, 5.930499050746588e-07), (np.array([1.0, 0.0, 0]), np.array([1.0, 0.0, 0.0]), 2000, 318.3098861837907), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 2000, 101371.86957712633), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0, 0, 0.0]), 2000, 0.00018886808182653578), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0, 0, 0.0]), 2000, 2.0255393314603194e-87)])
    def test_pdf_accuracy(self, x, mu, kappa, reference):
        if False:
            return 10
        pdf = vonmises_fisher(mu, kappa).pdf(x)
        assert_allclose(pdf, reference, rtol=1e-13)

    @pytest.mark.parametrize('x, mu, kappa, reference', [(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0001, -2.5309242486359573), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 0.0001, -2.5310242486359575), (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 2.767293119578746), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 100, -97.23270688042125), (np.array([1.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0]), 2000, -14.337987284534103), (np.array([1.0, 0.0, 0]), np.array([1.0, 0.0, 0.0]), 2000, 5.763025393132737), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 2000, 11.526550911307156), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0, 0, 0.0]), 2000, -8.574461766359684), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0, 0, 0.0]), 2000, -199.61906708886113)])
    def test_logpdf_accuracy(self, x, mu, kappa, reference):
        if False:
            i = 10
            return i + 15
        logpdf = vonmises_fisher(mu, kappa).logpdf(x)
        assert_allclose(logpdf, reference, rtol=1e-14)

    @pytest.mark.parametrize('dim, kappa, reference', [(3, 0.0001, 2.531024245302624), (3, 100, -1.7672931195787458), (5, 5000, -11.359032310024453), (8, 1, 3.4189526482545527)])
    def test_entropy_accuracy(self, dim, kappa, reference):
        if False:
            i = 10
            return i + 15
        mu = np.full((dim,), 1 / np.sqrt(dim))
        entropy = vonmises_fisher(mu, kappa).entropy()
        assert_allclose(entropy, reference, rtol=2e-14)

    @pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
    def test_broadcasting(self, method):
        if False:
            i = 10
            return i + 15
        testshape = (2, 2)
        rng = np.random.default_rng(2777937887058094419)
        x = uniform_direction(3).rvs(testshape, random_state=rng)
        mu = np.full((3,), 1 / np.sqrt(3))
        kappa = 5
        result_all = method(x, mu, kappa)
        assert result_all.shape == testshape
        for i in range(testshape[0]):
            for j in range(testshape[1]):
                current_val = method(x[i, j, :], mu, kappa)
                assert_allclose(current_val, result_all[i, j], rtol=1e-15)

    def test_vs_vonmises_2d(self):
        if False:
            return 10
        rng = np.random.default_rng(2777937887058094419)
        mu = np.array([0, 1])
        mu_angle = np.arctan2(mu[1], mu[0])
        kappa = 20
        vmf = vonmises_fisher(mu, kappa)
        vonmises_dist = vonmises(loc=mu_angle, kappa=kappa)
        vectors = uniform_direction(2).rvs(10, random_state=rng)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        assert_allclose(vonmises_dist.entropy(), vmf.entropy())
        assert_allclose(vonmises_dist.pdf(angles), vmf.pdf(vectors))
        assert_allclose(vonmises_dist.logpdf(angles), vmf.logpdf(vectors))

    @pytest.mark.parametrize('dim', [2, 3, 6])
    @pytest.mark.parametrize('kappa, mu_tol, kappa_tol', [(1, 0.05, 0.05), (10, 0.01, 0.01), (100, 0.005, 0.02), (1000, 0.001, 0.02)])
    def test_fit_accuracy(self, dim, kappa, mu_tol, kappa_tol):
        if False:
            for i in range(10):
                print('nop')
        mu = np.full((dim,), 1 / np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, kappa)
        rng = np.random.default_rng(2777937887058094419)
        n_samples = 10000
        samples = vmf_dist.rvs(n_samples, random_state=rng)
        (mu_fit, kappa_fit) = vonmises_fisher.fit(samples)
        angular_error = np.arccos(mu.dot(mu_fit))
        assert_allclose(angular_error, 0.0, atol=mu_tol, rtol=0)
        assert_allclose(kappa, kappa_fit, rtol=kappa_tol)

    def test_fit_error_one_dimensional_data(self):
        if False:
            while True:
                i = 10
        x = np.zeros((3,))
        msg = "'x' must be two dimensional."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_fit_error_unnormalized_data(self):
        if False:
            print('Hello World!')
        x = np.ones((3, 3))
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_frozen_distribution(self):
        if False:
            return 10
        mu = np.array([0, 0, 1])
        kappa = 5
        frozen = vonmises_fisher(mu, kappa)
        frozen_seed = vonmises_fisher(mu, kappa, seed=514)
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = vonmises_fisher.rvs(mu, kappa, random_state=514)
        rvs3 = frozen_seed.rvs()
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

class TestDirichletMultinomial:

    @classmethod
    def get_params(self, m):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, size=2)
        x = rng.integers(1, 20, size=(m, 2))
        n = x.sum(axis=-1)
        return (rng, m, alpha, n, x)

    def test_frozen(self):
        if False:
            return 10
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)
        d = dirichlet_multinomial(alpha, n)
        assert_equal(d.logpmf(x), dirichlet_multinomial.logpmf(x, alpha, n))
        assert_equal(d.pmf(x), dirichlet_multinomial.pmf(x, alpha, n))
        assert_equal(d.mean(), dirichlet_multinomial.mean(alpha, n))
        assert_equal(d.var(), dirichlet_multinomial.var(alpha, n))
        assert_equal(d.cov(), dirichlet_multinomial.cov(alpha, n))

    def test_pmf_logpmf_against_R(self):
        if False:
            while True:
                i = 10
        x = np.array([1, 2, 3])
        n = np.sum(x)
        alpha = np.array([3, 4, 5])
        res = dirichlet_multinomial.pmf(x, alpha, n)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 0.08484162895927638
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))
        assert res.shape == logres.shape == ()
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)
        res = dirichlet_multinomial(alpha, n).pmf(x)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 3.65409306285992e-16
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))

    def test_pmf_logpmf_support(self):
        if False:
            while True:
                i = 10
        (rng, m, alpha, n, x) = self.get_params(1)
        n += 1
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x), 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x), -np.inf)
        (rng, m, alpha, n, x) = self.get_params(10)
        i = rng.random(size=10) > 0.5
        x[i] = np.round(x[i] * 2)
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x)[i], 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x)[i], -np.inf)
        assert np.all(dirichlet_multinomial(alpha, n).pmf(x)[~i] > 0)
        assert np.all(dirichlet_multinomial(alpha, n).logpmf(x)[~i] > -np.inf)

    def test_dimensionality_one(self):
        if False:
            print('Hello World!')
        n = 6
        alpha = [10]
        x = np.asarray([n])
        dist = dirichlet_multinomial(alpha, n)
        assert_equal(dist.pmf(x), 1)
        assert_equal(dist.pmf(x + 1), 0)
        assert_equal(dist.logpmf(x), 0)
        assert_equal(dist.logpmf(x + 1), -np.inf)
        assert_equal(dist.mean(), n)
        assert_equal(dist.var(), 0)
        assert_equal(dist.cov(), 0)

    @pytest.mark.parametrize('method_name', ['pmf', 'logpmf'])
    def test_against_betabinom_pmf(self, method_name):
        if False:
            for i in range(10):
                print('nop')
        (rng, m, alpha, n, x) = self.get_params(100)
        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)
        res = method(x)
        ref = ref_method(x.T[0])
        assert_allclose(res, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var'])
    def test_against_betabinom_moments(self, method_name):
        if False:
            print('Hello World!')
        (rng, m, alpha, n, x) = self.get_params(100)
        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)
        res = method()[:, 0]
        ref = ref_method()
        assert_allclose(res, ref)

    def test_moments(self):
        if False:
            for i in range(10):
                print('nop')
        message = 'Needs NumPy 1.22.0 for multinomial broadcasting'
        if Version(np.__version__) < Version('1.22.0'):
            pytest.skip(reason=message)
        rng = np.random.default_rng(28469824356873456)
        dim = 5
        n = rng.integers(1, 100)
        alpha = rng.random(size=dim) * 10
        dist = dirichlet_multinomial(alpha, n)
        m = 100000
        p = rng.dirichlet(alpha, size=m)
        x = rng.multinomial(n, p, size=m)
        assert_allclose(dist.mean(), np.mean(x, axis=0), rtol=0.005)
        assert_allclose(dist.var(), np.var(x, axis=0), rtol=0.01)
        assert dist.mean().shape == dist.var().shape == (dim,)
        cov = dist.cov()
        assert cov.shape == (dim, dim)
        assert_allclose(cov, np.cov(x.T), rtol=0.02)
        assert_equal(np.diag(cov), dist.var())
        assert np.all(scipy.linalg.eigh(cov)[0] > 0)

    def test_input_validation(self):
        if False:
            i = 10
            return i + 15
        x0 = np.array([1, 2, 3])
        n0 = np.sum(x0)
        alpha0 = np.array([3, 4, 5])
        text = '`x` must contain only non-negative integers.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf([1, -1, 3], alpha0, n0)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf([1, 2.1, 3], alpha0, n0)
        text = '`alpha` must contain only positive values.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, [3, 0, 4], n0)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, [3, -1, 4], n0)
        text = '`n` must be a positive integer.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, alpha0, 49.1)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, alpha0, 0)
        x = np.array([1, 2, 3, 4])
        alpha = np.array([3, 4, 5])
        text = '`x` and `alpha` must be broadcastable.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x, alpha, x.sum())

    @pytest.mark.parametrize('method', ['pmf', 'logpmf'])
    def test_broadcasting_pmf(self, method):
        if False:
            i = 10
            return i + 15
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        n = np.array([[6], [7], [8]])
        x = np.array([[1, 2, 3], [2, 2, 3]]).reshape((2, 1, 1, 3))
        method = getattr(dirichlet_multinomial, method)
        res = method(x, alpha, n)
        assert res.shape == (2, 3, 4)
        for i in range(len(x)):
            for j in range(len(n)):
                for k in range(len(alpha)):
                    res_ijk = res[i, j, k]
                    ref = method(x[i].squeeze(), alpha[k].squeeze(), n[j].squeeze())
                    assert_allclose(res_ijk, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var', 'cov'])
    def test_broadcasting_moments(self, method_name):
        if False:
            i = 10
            return i + 15
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        n = np.array([[6], [7], [8]])
        method = getattr(dirichlet_multinomial, method_name)
        res = method(alpha, n)
        assert res.shape == (3, 4, 3) if method_name != 'cov' else (3, 4, 3, 3)
        for j in range(len(n)):
            for k in range(len(alpha)):
                res_ijk = res[j, k]
                ref = method(alpha[k].squeeze(), n[j].squeeze())
                assert_allclose(res_ijk, ref)