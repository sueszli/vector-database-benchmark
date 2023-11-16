import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky, _compute_precision_cholesky, _estimate_gaussian_covariances_diag, _estimate_gaussian_covariances_full, _estimate_gaussian_covariances_spherical, _estimate_gaussian_covariances_tied, _estimate_gaussian_parameters
from sklearn.utils._testing import assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal, ignore_warnings
from sklearn.utils.extmath import fast_logdet
COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']

def generate_data(n_samples, n_features, weights, means, precisions, covariance_type):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = []
    if covariance_type == 'spherical':
        for (_, (w, m, c)) in enumerate(zip(weights, means, precisions['spherical'])):
            X.append(rng.multivariate_normal(m, c * np.eye(n_features), int(np.round(w * n_samples))))
    if covariance_type == 'diag':
        for (_, (w, m, c)) in enumerate(zip(weights, means, precisions['diag'])):
            X.append(rng.multivariate_normal(m, np.diag(c), int(np.round(w * n_samples))))
    if covariance_type == 'tied':
        for (_, (w, m)) in enumerate(zip(weights, means)):
            X.append(rng.multivariate_normal(m, precisions['tied'], int(np.round(w * n_samples))))
    if covariance_type == 'full':
        for (_, (w, m, c)) in enumerate(zip(weights, means, precisions['full'])):
            X.append(rng.multivariate_normal(m, c, int(np.round(w * n_samples))))
    X = np.vstack(X)
    return X

class RandomData:

    def __init__(self, rng, n_samples=200, n_components=2, n_features=2, scale=50):
        if False:
            while True:
                i = 10
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features
        self.weights = rng.rand(n_components)
        self.weights = self.weights / self.weights.sum()
        self.means = rng.rand(n_components, n_features) * scale
        self.covariances = {'spherical': 0.5 + rng.rand(n_components), 'diag': (0.5 + rng.rand(n_components, n_features)) ** 2, 'tied': make_spd_matrix(n_features, random_state=rng), 'full': np.array([make_spd_matrix(n_features, random_state=rng) * 0.5 for _ in range(n_components)])}
        self.precisions = {'spherical': 1.0 / self.covariances['spherical'], 'diag': 1.0 / self.covariances['diag'], 'tied': linalg.inv(self.covariances['tied']), 'full': np.array([linalg.inv(covariance) for covariance in self.covariances['full']])}
        self.X = dict(zip(COVARIANCE_TYPE, [generate_data(n_samples, n_features, self.weights, self.means, self.covariances, covar_type) for covar_type in COVARIANCE_TYPE]))
        self.Y = np.hstack([np.full(int(np.round(w * n_samples)), k, dtype=int) for (k, w) in enumerate(self.weights)])

def test_gaussian_mixture_attributes():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    (n_components, tol, n_init, max_iter, reg_covar) = (2, 0.0001, 3, 30, 0.1)
    (covariance_type, init_params) = ('full', 'random')
    gmm = GaussianMixture(n_components=n_components, tol=tol, n_init=n_init, max_iter=max_iter, reg_covar=reg_covar, covariance_type=covariance_type, init_params=init_params).fit(X)
    assert gmm.n_components == n_components
    assert gmm.covariance_type == covariance_type
    assert gmm.tol == tol
    assert gmm.reg_covar == reg_covar
    assert gmm.max_iter == max_iter
    assert gmm.n_init == n_init
    assert gmm.init_params == init_params

def test_check_weights():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    g = GaussianMixture(n_components=n_components)
    weights_bad_shape = rng.rand(n_components, 1)
    g.weights_init = weights_bad_shape
    msg = re.escape(f"The parameter 'weights' should have the shape of ({n_components},), but got {str(weights_bad_shape.shape)}")
    with pytest.raises(ValueError, match=msg):
        g.fit(X)
    weights_bad_range = rng.rand(n_components) + 1
    g.weights_init = weights_bad_range
    msg = re.escape(f"The parameter 'weights' should be in the range [0, 1], but got max value {np.min(weights_bad_range):.5f}, min value {np.max(weights_bad_range):.5f}")
    with pytest.raises(ValueError, match=msg):
        g.fit(X)
    weights_bad_norm = rng.rand(n_components)
    weights_bad_norm = weights_bad_norm / (weights_bad_norm.sum() + 1)
    g.weights_init = weights_bad_norm
    msg = re.escape(f"The parameter 'weights' should be normalized, but got sum(weights) = {np.sum(weights_bad_norm):.5f}")
    with pytest.raises(ValueError, match=msg):
        g.fit(X)
    weights = rand_data.weights
    g = GaussianMixture(weights_init=weights, n_components=n_components)
    g.fit(X)
    assert_array_equal(weights, g.weights_init)

def test_check_means():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    (n_components, n_features) = (rand_data.n_components, rand_data.n_features)
    X = rand_data.X['full']
    g = GaussianMixture(n_components=n_components)
    means_bad_shape = rng.rand(n_components + 1, n_features)
    g.means_init = means_bad_shape
    msg = "The parameter 'means' should have the shape of "
    with pytest.raises(ValueError, match=msg):
        g.fit(X)
    means = rand_data.means
    g.means_init = means
    g.fit(X)
    assert_array_equal(means, g.means_init)

def test_check_precisions():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    (n_components, n_features) = (rand_data.n_components, rand_data.n_features)
    precisions_bad_shape = {'full': np.ones((n_components + 1, n_features, n_features)), 'tied': np.ones((n_features + 1, n_features + 1)), 'diag': np.ones((n_components + 1, n_features)), 'spherical': np.ones(n_components + 1)}
    precisions_not_pos = np.ones((n_components, n_features, n_features))
    precisions_not_pos[0] = np.eye(n_features)
    precisions_not_pos[0, 0, 0] = -1.0
    precisions_not_positive = {'full': precisions_not_pos, 'tied': precisions_not_pos[0], 'diag': np.full((n_components, n_features), -1.0), 'spherical': np.full(n_components, -1.0)}
    not_positive_errors = {'full': 'symmetric, positive-definite', 'tied': 'symmetric, positive-definite', 'diag': 'positive', 'spherical': 'positive'}
    for covar_type in COVARIANCE_TYPE:
        X = RandomData(rng).X[covar_type]
        g = GaussianMixture(n_components=n_components, covariance_type=covar_type, random_state=rng)
        g.precisions_init = precisions_bad_shape[covar_type]
        msg = f"The parameter '{covar_type} precision' should have the shape of"
        with pytest.raises(ValueError, match=msg):
            g.fit(X)
        g.precisions_init = precisions_not_positive[covar_type]
        msg = f"'{covar_type} precision' should be {not_positive_errors[covar_type]}"
        with pytest.raises(ValueError, match=msg):
            g.fit(X)
        g.precisions_init = rand_data.precisions[covar_type]
        g.fit(X)
        assert_array_equal(rand_data.precisions[covar_type], g.precisions_init)

def test_suffstat_sk_full():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (500, 2)
    X = rng.rand(n_samples, n_features)
    resp = rng.rand(n_samples, 1)
    X_resp = np.sqrt(resp) * X
    nk = np.array([n_samples])
    xk = np.zeros((1, n_features))
    covars_pred = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    ecov = EmpiricalCovariance(assume_centered=True)
    ecov.fit(X_resp)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='spectral'), 0)
    precs_chol_pred = _compute_precision_cholesky(covars_pred, 'full')
    precs_pred = np.array([np.dot(prec, prec.T) for prec in precs_chol_pred])
    precs_est = np.array([linalg.inv(cov) for cov in covars_pred])
    assert_array_almost_equal(precs_est, precs_pred)
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean(axis=0).reshape((1, -1))
    covars_pred = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    ecov = EmpiricalCovariance(assume_centered=False)
    ecov.fit(X)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='spectral'), 0)
    precs_chol_pred = _compute_precision_cholesky(covars_pred, 'full')
    precs_pred = np.array([np.dot(prec, prec.T) for prec in precs_chol_pred])
    precs_est = np.array([linalg.inv(cov) for cov in covars_pred])
    assert_array_almost_equal(precs_est, precs_pred)

def test_suffstat_sk_tied():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features, n_components) = (500, 2, 2)
    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    covars_pred_full = np.sum(nk[:, np.newaxis, np.newaxis] * covars_pred_full, 0) / n_samples
    covars_pred_tied = _estimate_gaussian_covariances_tied(resp, X, nk, xk, 0)
    ecov = EmpiricalCovariance()
    ecov.covariance_ = covars_pred_full
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='spectral'), 0)
    precs_chol_pred = _compute_precision_cholesky(covars_pred_tied, 'tied')
    precs_pred = np.dot(precs_chol_pred, precs_chol_pred.T)
    precs_est = linalg.inv(covars_pred_tied)
    assert_array_almost_equal(precs_est, precs_pred)

def test_suffstat_sk_diag():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features, n_components) = (500, 2, 2)
    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    covars_pred_diag = _estimate_gaussian_covariances_diag(resp, X, nk, xk, 0)
    ecov = EmpiricalCovariance()
    for (cov_full, cov_diag) in zip(covars_pred_full, covars_pred_diag):
        ecov.covariance_ = np.diag(np.diag(cov_full))
        cov_diag = np.diag(cov_diag)
        assert_almost_equal(ecov.error_norm(cov_diag, norm='frobenius'), 0)
        assert_almost_equal(ecov.error_norm(cov_diag, norm='spectral'), 0)
    precs_chol_pred = _compute_precision_cholesky(covars_pred_diag, 'diag')
    assert_almost_equal(covars_pred_diag, 1.0 / precs_chol_pred ** 2)

def test_gaussian_suffstat_sk_spherical():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (500, 2)
    X = rng.rand(n_samples, n_features)
    X = X - X.mean()
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean()
    covars_pred_spherical = _estimate_gaussian_covariances_spherical(resp, X, nk, xk, 0)
    covars_pred_spherical2 = np.dot(X.flatten().T, X.flatten()) / (n_features * n_samples)
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)
    precs_chol_pred = _compute_precision_cholesky(covars_pred_spherical, 'spherical')
    assert_almost_equal(covars_pred_spherical, 1.0 / precs_chol_pred ** 2)

def test_compute_log_det_cholesky():
    if False:
        print('Hello World!')
    n_features = 2
    rand_data = RandomData(np.random.RandomState(0))
    for covar_type in COVARIANCE_TYPE:
        covariance = rand_data.covariances[covar_type]
        if covar_type == 'full':
            predected_det = np.array([linalg.det(cov) for cov in covariance])
        elif covar_type == 'tied':
            predected_det = linalg.det(covariance)
        elif covar_type == 'diag':
            predected_det = np.array([np.prod(cov) for cov in covariance])
        elif covar_type == 'spherical':
            predected_det = covariance ** n_features
        expected_det = _compute_log_det_cholesky(_compute_precision_cholesky(covariance, covar_type), covar_type, n_features=n_features)
        assert_array_almost_equal(expected_det, -0.5 * np.log(predected_det))

def _naive_lmvnpdf_diag(X, means, covars):
    if False:
        while True:
            i = 10
    resp = np.empty((len(X), len(means)))
    stds = np.sqrt(covars)
    for (i, (mean, std)) in enumerate(zip(means, stds)):
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    return resp

def test_gaussian_mixture_log_probabilities():
    if False:
        while True:
            i = 10
    from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_samples = 500
    n_features = rand_data.n_features
    n_components = rand_data.n_components
    means = rand_data.means
    covars_diag = rng.rand(n_components, n_features)
    X = rng.rand(n_samples, n_features)
    log_prob_naive = _naive_lmvnpdf_diag(X, means, covars_diag)
    precs_full = np.array([np.diag(1.0 / np.sqrt(x)) for x in covars_diag])
    log_prob = _estimate_log_gaussian_prob(X, means, precs_full, 'full')
    assert_array_almost_equal(log_prob, log_prob_naive)
    precs_chol_diag = 1.0 / np.sqrt(covars_diag)
    log_prob = _estimate_log_gaussian_prob(X, means, precs_chol_diag, 'diag')
    assert_array_almost_equal(log_prob, log_prob_naive)
    covars_tied = np.array([x for x in covars_diag]).mean(axis=0)
    precs_tied = np.diag(np.sqrt(1.0 / covars_tied))
    log_prob_naive = _naive_lmvnpdf_diag(X, means, [covars_tied] * n_components)
    log_prob = _estimate_log_gaussian_prob(X, means, precs_tied, 'tied')
    assert_array_almost_equal(log_prob, log_prob_naive)
    covars_spherical = covars_diag.mean(axis=1)
    precs_spherical = 1.0 / np.sqrt(covars_diag.mean(axis=1))
    log_prob_naive = _naive_lmvnpdf_diag(X, means, [[k] * n_features for k in covars_spherical])
    log_prob = _estimate_log_gaussian_prob(X, means, precs_spherical, 'spherical')
    assert_array_almost_equal(log_prob, log_prob_naive)

def test_gaussian_mixture_estimate_log_prob_resp():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_samples = rand_data.n_samples
    n_features = rand_data.n_features
    n_components = rand_data.n_components
    X = rng.rand(n_samples, n_features)
    for covar_type in COVARIANCE_TYPE:
        weights = rand_data.weights
        means = rand_data.means
        precisions = rand_data.precisions[covar_type]
        g = GaussianMixture(n_components=n_components, random_state=rng, weights_init=weights, means_init=means, precisions_init=precisions, covariance_type=covar_type)
        g.fit(X)
        resp = g.predict_proba(X)
        assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))
        assert_array_equal(g.weights_init, weights)
        assert_array_equal(g.means_init, means)
        assert_array_equal(g.precisions_init, precisions)

def test_gaussian_mixture_predict_predict_proba():
    if False:
        return 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        Y = rand_data.Y
        g = GaussianMixture(n_components=rand_data.n_components, random_state=rng, weights_init=rand_data.weights, means_init=rand_data.means, precisions_init=rand_data.precisions[covar_type], covariance_type=covar_type)
        msg = "This GaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        with pytest.raises(NotFittedError, match=msg):
            g.predict(X)
        g.fit(X)
        Y_pred = g.predict(X)
        Y_pred_proba = g.predict_proba(X).argmax(axis=1)
        assert_array_equal(Y_pred, Y_pred_proba)
        assert adjusted_rand_score(Y, Y_pred) > 0.95

@pytest.mark.filterwarnings('ignore:.*did not converge.*')
@pytest.mark.parametrize('seed, max_iter, tol', [(0, 2, 1e-07), (1, 2, 0.1), (3, 300, 1e-07), (4, 300, 0.1)])
def test_gaussian_mixture_fit_predict(seed, max_iter, tol):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(seed)
    rand_data = RandomData(rng)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        Y = rand_data.Y
        g = GaussianMixture(n_components=rand_data.n_components, random_state=rng, weights_init=rand_data.weights, means_init=rand_data.means, precisions_init=rand_data.precisions[covar_type], covariance_type=covar_type, max_iter=max_iter, tol=tol)
        f = copy.deepcopy(g)
        Y_pred1 = f.fit(X).predict(X)
        Y_pred2 = g.fit_predict(X)
        assert_array_equal(Y_pred1, Y_pred2)
        assert adjusted_rand_score(Y, Y_pred2) > 0.95

def test_gaussian_mixture_fit_predict_n_init():
    if False:
        print('Hello World!')
    X = np.random.RandomState(0).randn(1000, 5)
    gm = GaussianMixture(n_components=5, n_init=5, random_state=0)
    y_pred1 = gm.fit_predict(X)
    y_pred2 = gm.predict(X)
    assert_array_equal(y_pred1, y_pred2)

def test_gaussian_mixture_fit():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_features = rand_data.n_features
    n_components = rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = GaussianMixture(n_components=n_components, n_init=20, reg_covar=0, random_state=rng, covariance_type=covar_type)
        g.fit(X)
        assert_allclose(np.sort(g.weights_), np.sort(rand_data.weights), rtol=0.1, atol=0.01)
        arg_idx1 = g.means_[:, 0].argsort()
        arg_idx2 = rand_data.means[:, 0].argsort()
        assert_allclose(g.means_[arg_idx1], rand_data.means[arg_idx2], rtol=0.1, atol=0.01)
        if covar_type == 'full':
            prec_pred = g.precisions_
            prec_test = rand_data.precisions['full']
        elif covar_type == 'tied':
            prec_pred = np.array([g.precisions_] * n_components)
            prec_test = np.array([rand_data.precisions['tied']] * n_components)
        elif covar_type == 'spherical':
            prec_pred = np.array([np.eye(n_features) * c for c in g.precisions_])
            prec_test = np.array([np.eye(n_features) * c for c in rand_data.precisions['spherical']])
        elif covar_type == 'diag':
            prec_pred = np.array([np.diag(d) for d in g.precisions_])
            prec_test = np.array([np.diag(d) for d in rand_data.precisions['diag']])
        arg_idx1 = np.trace(prec_pred, axis1=1, axis2=2).argsort()
        arg_idx2 = np.trace(prec_test, axis1=1, axis2=2).argsort()
        for (k, h) in zip(arg_idx1, arg_idx2):
            ecov = EmpiricalCovariance()
            ecov.covariance_ = prec_test[h]
            assert_allclose(ecov.error_norm(prec_pred[k]), 0, atol=0.15)

def test_gaussian_mixture_fit_best_params():
    if False:
        return 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    n_init = 10
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = GaussianMixture(n_components=n_components, n_init=1, reg_covar=0, random_state=rng, covariance_type=covar_type)
        ll = []
        for _ in range(n_init):
            g.fit(X)
            ll.append(g.score(X))
        ll = np.array(ll)
        g_best = GaussianMixture(n_components=n_components, n_init=n_init, reg_covar=0, random_state=rng, covariance_type=covar_type)
        g_best.fit(X)
        assert_almost_equal(ll.min(), g_best.score(X))

def test_gaussian_mixture_fit_convergence_warning():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=1)
    n_components = rand_data.n_components
    max_iter = 1
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = GaussianMixture(n_components=n_components, n_init=1, max_iter=max_iter, reg_covar=0, random_state=rng, covariance_type=covar_type)
        msg = f'Initialization {max_iter} did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.'
        with pytest.warns(ConvergenceWarning, match=msg):
            g.fit(X)

def test_multiple_init():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    (n_samples, n_features, n_components) = (50, 5, 2)
    X = rng.randn(n_samples, n_features)
    for cv_type in COVARIANCE_TYPE:
        train1 = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=0).fit(X).score(X)
        train2 = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=0, n_init=5).fit(X).score(X)
        assert train2 >= train1

def test_gaussian_mixture_n_parameters():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features, n_components) = (50, 5, 2)
    X = rng.randn(n_samples, n_features)
    n_params = {'spherical': 13, 'diag': 21, 'tied': 26, 'full': 41}
    for cv_type in COVARIANCE_TYPE:
        g = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=rng).fit(X)
        assert g._n_parameters() == n_params[cv_type]

def test_bic_1d_1component():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    (n_samples, n_dim, n_components) = (100, 1, 1)
    X = rng.randn(n_samples, n_dim)
    bic_full = GaussianMixture(n_components=n_components, covariance_type='full', random_state=rng).fit(X).bic(X)
    for covariance_type in ['tied', 'diag', 'spherical']:
        bic = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=rng).fit(X).bic(X)
        assert_almost_equal(bic_full, bic)

def test_gaussian_mixture_aic_bic():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    (n_samples, n_features, n_components) = (50, 3, 2)
    X = rng.randn(n_samples, n_features)
    sgh = 0.5 * (fast_logdet(np.cov(X.T, bias=1)) + n_features * (1 + np.log(2 * np.pi)))
    for cv_type in COVARIANCE_TYPE:
        g = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=rng, max_iter=200)
        g.fit(X)
        aic = 2 * n_samples * sgh + 2 * g._n_parameters()
        bic = 2 * n_samples * sgh + np.log(n_samples) * g._n_parameters()
        bound = n_features / np.sqrt(n_samples)
        assert (g.aic(X) - aic) / n_samples < bound
        assert (g.bic(X) - bic) / n_samples < bound

def test_gaussian_mixture_verbose():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = GaussianMixture(n_components=n_components, n_init=1, reg_covar=0, random_state=rng, covariance_type=covar_type, verbose=1)
        h = GaussianMixture(n_components=n_components, n_init=1, reg_covar=0, random_state=rng, covariance_type=covar_type, verbose=2)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            g.fit(X)
            h.fit(X)
        finally:
            sys.stdout = old_stdout

@pytest.mark.filterwarnings('ignore:.*did not converge.*')
@pytest.mark.parametrize('seed', (0, 1, 2))
def test_warm_start(seed):
    if False:
        print('Hello World!')
    random_state = seed
    rng = np.random.RandomState(random_state)
    (n_samples, n_features, n_components) = (500, 2, 2)
    X = rng.rand(n_samples, n_features)
    g = GaussianMixture(n_components=n_components, n_init=1, max_iter=2, reg_covar=0, random_state=random_state, warm_start=False)
    h = GaussianMixture(n_components=n_components, n_init=1, max_iter=1, reg_covar=0, random_state=random_state, warm_start=True)
    g.fit(X)
    score1 = h.fit(X).score(X)
    score2 = h.fit(X).score(X)
    assert_almost_equal(g.weights_, h.weights_)
    assert_almost_equal(g.means_, h.means_)
    assert_almost_equal(g.precisions_, h.precisions_)
    assert score2 > score1
    g = GaussianMixture(n_components=n_components, n_init=1, max_iter=5, reg_covar=0, random_state=random_state, warm_start=False, tol=1e-06)
    h = GaussianMixture(n_components=n_components, n_init=1, max_iter=5, reg_covar=0, random_state=random_state, warm_start=True, tol=1e-06)
    g.fit(X)
    assert not g.converged_
    h.fit(X)
    for _ in range(1000):
        h.fit(X)
        if h.converged_:
            break
    assert h.converged_

@ignore_warnings(category=ConvergenceWarning)
def test_convergence_detected_with_warm_start():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    for max_iter in (1, 2, 50):
        gmm = GaussianMixture(n_components=n_components, warm_start=True, max_iter=max_iter, random_state=rng)
        for _ in range(100):
            gmm.fit(X)
            if gmm.converged_:
                break
        assert gmm.converged_
        assert max_iter >= gmm.n_iter_

def test_score():
    if False:
        print('Hello World!')
    covar_type = 'full'
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X[covar_type]
    gmm1 = GaussianMixture(n_components=n_components, n_init=1, max_iter=1, reg_covar=0, random_state=rng, covariance_type=covar_type)
    msg = "This GaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    with pytest.raises(NotFittedError, match=msg):
        gmm1.score(X)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gmm1.fit(X)
    gmm_score = gmm1.score(X)
    gmm_score_proba = gmm1.score_samples(X).mean()
    assert_almost_equal(gmm_score, gmm_score_proba)
    gmm2 = GaussianMixture(n_components=n_components, n_init=1, reg_covar=0, random_state=rng, covariance_type=covar_type).fit(X)
    assert gmm2.score(X) > gmm1.score(X)

def test_score_samples():
    if False:
        i = 10
        return i + 15
    covar_type = 'full'
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X[covar_type]
    gmm = GaussianMixture(n_components=n_components, n_init=1, reg_covar=0, random_state=rng, covariance_type=covar_type)
    msg = "This GaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    with pytest.raises(NotFittedError, match=msg):
        gmm.score_samples(X)
    gmm_score_samples = gmm.fit(X).score_samples(X)
    assert gmm_score_samples.shape[0] == rand_data.n_samples

def test_monotonic_likelihood():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, reg_covar=0, warm_start=True, max_iter=1, random_state=rng, tol=1e-07)
        current_log_likelihood = -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            for _ in range(600):
                prev_log_likelihood = current_log_likelihood
                current_log_likelihood = gmm.fit(X).score(X)
                assert current_log_likelihood >= prev_log_likelihood
                if gmm.converged_:
                    break
            assert gmm.converged_

def test_regularisation():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 5)
    X = np.vstack((np.ones((n_samples // 2, n_features)), np.zeros((n_samples // 2, n_features))))
    for covar_type in COVARIANCE_TYPE:
        gmm = GaussianMixture(n_components=n_samples, reg_covar=0, covariance_type=covar_type, random_state=rng)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            msg = re.escape('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, or increase reg_covar.')
            with pytest.raises(ValueError, match=msg):
                gmm.fit(X)
            gmm.set_params(reg_covar=1e-06).fit(X)

def test_property():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, random_state=rng, n_init=5)
        gmm.fit(X)
        if covar_type == 'full':
            for (prec, covar) in zip(gmm.precisions_, gmm.covariances_):
                assert_array_almost_equal(linalg.inv(prec), covar)
        elif covar_type == 'tied':
            assert_array_almost_equal(linalg.inv(gmm.precisions_), gmm.covariances_)
        else:
            assert_array_almost_equal(gmm.precisions_, 1.0 / gmm.covariances_)

def test_sample():
    if False:
        return 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7, n_components=3)
    (n_features, n_components) = (rand_data.n_features, rand_data.n_components)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, random_state=rng)
        msg = 'This GaussianMixture instance is not fitted'
        with pytest.raises(NotFittedError, match=msg):
            gmm.sample(0)
        gmm.fit(X)
        msg = "Invalid value for 'n_samples'"
        with pytest.raises(ValueError, match=msg):
            gmm.sample(0)
        n_samples = 20000
        (X_s, y_s) = gmm.sample(n_samples)
        for k in range(n_components):
            if covar_type == 'full':
                assert_array_almost_equal(gmm.covariances_[k], np.cov(X_s[y_s == k].T), decimal=1)
            elif covar_type == 'tied':
                assert_array_almost_equal(gmm.covariances_, np.cov(X_s[y_s == k].T), decimal=1)
            elif covar_type == 'diag':
                assert_array_almost_equal(gmm.covariances_[k], np.diag(np.cov(X_s[y_s == k].T)), decimal=1)
            else:
                assert_array_almost_equal(gmm.covariances_[k], np.var(X_s[y_s == k] - gmm.means_[k]), decimal=1)
        means_s = np.array([np.mean(X_s[y_s == k], 0) for k in range(n_components)])
        assert_array_almost_equal(gmm.means_, means_s, decimal=1)
        assert X_s.shape == (n_samples, n_features)
        for sample_size in range(1, 100):
            (X_s, _) = gmm.sample(sample_size)
            assert X_s.shape == (sample_size, n_features)

@ignore_warnings(category=ConvergenceWarning)
def test_init():
    if False:
        while True:
            i = 10
    for random_state in range(15):
        rand_data = RandomData(np.random.RandomState(random_state), n_samples=50, scale=1)
        n_components = rand_data.n_components
        X = rand_data.X['full']
        gmm1 = GaussianMixture(n_components=n_components, n_init=1, max_iter=1, random_state=random_state).fit(X)
        gmm2 = GaussianMixture(n_components=n_components, n_init=10, max_iter=1, random_state=random_state).fit(X)
        assert gmm2.lower_bound_ >= gmm1.lower_bound_

def test_gaussian_mixture_setting_best_params():
    if False:
        i = 10
        return i + 15
    "`GaussianMixture`'s best_parameters, `n_iter_` and `lower_bound_`\n    must be set appropriately in the case of divergence.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/18216\n    "
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = rnd.uniform(size=(n_samples, 3))
    means_init = np.array([[0.670637869618158, 0.21038256107384043, 0.12892629765485303], [0.09394051075844147, 0.5759464955561779, 0.929296197576212], [0.5033230372781258, 0.9569852381759425, 0.08654043447295741], [0.18578301420435747, 0.5531158970919143, 0.19388943970532435], [0.4548589928173794, 0.35182513658825276, 0.568146063202464], [0.609279894978321, 0.7929063819678847, 0.9620097270828052]])
    precisions_init = np.array([999999.999604483, 999999.9990869573, 553.7603944542167, 204.78596008931834, 15.867423501783637, 85.4595728389735])
    weights_init = [0.03333333333333341, 0.03333333333333341, 0.06666666666666674, 0.06666666666666674, 0.7000000000000001, 0.10000000000000007]
    gmm = GaussianMixture(covariance_type='spherical', reg_covar=0, means_init=means_init, weights_init=weights_init, random_state=rnd, n_components=len(weights_init), precisions_init=precisions_init, max_iter=1)
    gmm.fit(X)
    assert not gmm.converged_
    for attr in ['weights_', 'means_', 'covariances_', 'precisions_cholesky_', 'n_iter_', 'lower_bound_']:
        assert hasattr(gmm, attr)

@pytest.mark.parametrize('init_params', ['random', 'random_from_data', 'k-means++', 'kmeans'])
def test_init_means_not_duplicated(init_params, global_random_seed):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(global_random_seed)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    gmm = GaussianMixture(n_components=n_components, init_params=init_params, random_state=rng, max_iter=0)
    gmm.fit(X)
    means = gmm.means_
    for (i_mean, j_mean) in itertools.combinations(means, r=2):
        assert not np.allclose(i_mean, j_mean)

@pytest.mark.parametrize('init_params', ['random', 'random_from_data', 'k-means++', 'kmeans'])
def test_means_for_all_inits(init_params, global_random_seed):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(global_random_seed)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    gmm = GaussianMixture(n_components=n_components, init_params=init_params, random_state=rng)
    gmm.fit(X)
    assert gmm.means_.shape == (n_components, X.shape[1])
    assert np.all(X.min(axis=0) <= gmm.means_)
    assert np.all(gmm.means_ <= X.max(axis=0))
    assert gmm.converged_

def test_max_iter_zero():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    means_init = [[20, 30], [30, 25]]
    gmm = GaussianMixture(n_components=n_components, random_state=rng, means_init=means_init, tol=1e-06, max_iter=0)
    gmm.fit(X)
    assert_allclose(gmm.means_, means_init)

def test_gaussian_mixture_precisions_init_diag():
    if False:
        print('Hello World!')
    'Check that we properly initialize `precision_cholesky_` when we manually\n    provide the precision matrix.\n\n    In this regard, we check the consistency between estimating the precision\n    matrix and providing the same precision matrix as initialization. It should\n    lead to the same results with the same number of iterations.\n\n    If the initialization is wrong then the number of iterations will increase.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/16944\n    '
    n_samples = 300
    rng = np.random.RandomState(0)
    shifted_gaussian = rng.randn(n_samples, 2) + np.array([20, 20])
    C = np.array([[0.0, -0.7], [3.5, 0.7]])
    stretched_gaussian = np.dot(rng.randn(n_samples, 2), C)
    X = np.vstack([shifted_gaussian, stretched_gaussian])
    (n_components, covariance_type, reg_covar, random_state) = (2, 'diag', 1e-06, 0)
    resp = np.zeros((X.shape[0], n_components))
    label = KMeans(n_clusters=n_components, n_init=1, random_state=random_state).fit(X).labels_
    resp[np.arange(X.shape[0]), label] = 1
    (_, _, covariance) = _estimate_gaussian_parameters(X, resp, reg_covar=reg_covar, covariance_type=covariance_type)
    precisions_init = 1 / covariance
    gm_with_init = GaussianMixture(n_components=n_components, covariance_type=covariance_type, reg_covar=reg_covar, precisions_init=precisions_init, random_state=random_state).fit(X)
    gm_without_init = GaussianMixture(n_components=n_components, covariance_type=covariance_type, reg_covar=reg_covar, random_state=random_state).fit(X)
    assert gm_without_init.n_iter_ == gm_with_init.n_iter_
    assert_allclose(gm_with_init.precisions_cholesky_, gm_without_init.precisions_cholesky_)

def _generate_data(seed, n_samples, n_features, n_components):
    if False:
        i = 10
        return i + 15
    'Randomly generate samples and responsibilities.'
    rs = np.random.RandomState(seed)
    X = rs.random_sample((n_samples, n_features))
    resp = rs.random_sample((n_samples, n_components))
    resp /= resp.sum(axis=1)[:, np.newaxis]
    return (X, resp)

def _calculate_precisions(X, resp, covariance_type):
    if False:
        print('Hello World!')
    'Calculate precision matrix of X and its Cholesky decomposition\n    for the given covariance type.\n    '
    reg_covar = 1e-06
    (weights, means, covariances) = _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type)
    precisions_cholesky = _compute_precision_cholesky(covariances, covariance_type)
    (_, n_components) = resp.shape
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    params = (weights, means, covariances, precisions_cholesky)
    gmm._set_parameters(params)
    return (gmm.precisions_, gmm.precisions_cholesky_)

@pytest.mark.parametrize('covariance_type', COVARIANCE_TYPE)
def test_gaussian_mixture_precisions_init(covariance_type, global_random_seed):
    if False:
        return 10
    'Non-regression test for #26415.'
    (X, resp) = _generate_data(seed=global_random_seed, n_samples=100, n_features=3, n_components=4)
    (precisions_init, desired_precisions_cholesky) = _calculate_precisions(X, resp, covariance_type)
    gmm = GaussianMixture(covariance_type=covariance_type, precisions_init=precisions_init)
    gmm._initialize(X, resp)
    actual_precisions_cholesky = gmm.precisions_cholesky_
    assert_allclose(actual_precisions_cholesky, desired_precisions_cholesky)

def test_gaussian_mixture_single_component_stable():
    if False:
        for i in range(10):
            print('nop')
    '\n    Non-regression test for #23032 ensuring 1-component GM works on only a\n    few samples.\n    '
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(np.zeros(2), np.identity(2), size=3)
    gm = GaussianMixture(n_components=1)
    gm.fit(X).sample()

def test_gaussian_mixture_all_init_does_not_estimate_gaussian_parameters(monkeypatch, global_random_seed):
    if False:
        return 10
    'When all init parameters are provided, the Gaussian parameters\n    are not estimated.\n\n    Non-regression test for gh-26015.\n    '
    mock = Mock(side_effect=_estimate_gaussian_parameters)
    monkeypatch.setattr(sklearn.mixture._gaussian_mixture, '_estimate_gaussian_parameters', mock)
    rng = np.random.RandomState(global_random_seed)
    rand_data = RandomData(rng)
    gm = GaussianMixture(n_components=rand_data.n_components, weights_init=rand_data.weights, means_init=rand_data.means, precisions_init=rand_data.precisions['full'], random_state=rng)
    gm.fit(rand_data.X['full'])
    assert mock.call_count == gm.n_iter_