import copy
import numpy as np
import pytest
from scipy.special import gammaln
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._bayesian_mixture import _log_dirichlet_norm, _log_wishart_norm
from sklearn.mixture.tests.test_gaussian_mixture import RandomData
from sklearn.utils._testing import assert_almost_equal, assert_array_equal, ignore_warnings
COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']
PRIOR_TYPE = ['dirichlet_process', 'dirichlet_distribution']

def test_log_dirichlet_norm():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    weight_concentration = rng.rand(2)
    expected_norm = gammaln(np.sum(weight_concentration)) - np.sum(gammaln(weight_concentration))
    predected_norm = _log_dirichlet_norm(weight_concentration)
    assert_almost_equal(expected_norm, predected_norm)

def test_log_wishart_norm():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_components, n_features) = (5, 2)
    degrees_of_freedom = np.abs(rng.rand(n_components)) + 1.0
    log_det_precisions_chol = n_features * np.log(range(2, 2 + n_components))
    expected_norm = np.empty(5)
    for (k, (degrees_of_freedom_k, log_det_k)) in enumerate(zip(degrees_of_freedom, log_det_precisions_chol)):
        expected_norm[k] = -(degrees_of_freedom_k * (log_det_k + 0.5 * n_features * np.log(2.0)) + np.sum(gammaln(0.5 * (degrees_of_freedom_k - np.arange(0, n_features)[:, np.newaxis])), 0)).item()
    predected_norm = _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features)
    assert_almost_equal(expected_norm, predected_norm)

def test_bayesian_mixture_weights_prior_initialisation():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_components, n_features) = (10, 5, 2)
    X = rng.rand(n_samples, n_features)
    weight_concentration_prior = rng.rand()
    bgmm = BayesianGaussianMixture(weight_concentration_prior=weight_concentration_prior, random_state=rng).fit(X)
    assert_almost_equal(weight_concentration_prior, bgmm.weight_concentration_prior_)
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=rng).fit(X)
    assert_almost_equal(1.0 / n_components, bgmm.weight_concentration_prior_)

def test_bayesian_mixture_mean_prior_initialisation():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    (n_samples, n_components, n_features) = (10, 3, 2)
    X = rng.rand(n_samples, n_features)
    mean_precision_prior = rng.rand()
    bgmm = BayesianGaussianMixture(mean_precision_prior=mean_precision_prior, random_state=rng).fit(X)
    assert_almost_equal(mean_precision_prior, bgmm.mean_precision_prior_)
    bgmm = BayesianGaussianMixture(random_state=rng).fit(X)
    assert_almost_equal(1.0, bgmm.mean_precision_prior_)
    mean_prior = rng.rand(n_features)
    bgmm = BayesianGaussianMixture(n_components=n_components, mean_prior=mean_prior, random_state=rng).fit(X)
    assert_almost_equal(mean_prior, bgmm.mean_prior_)
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=rng).fit(X)
    assert_almost_equal(X.mean(axis=0), bgmm.mean_prior_)

def test_bayesian_mixture_precisions_prior_initialisation():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 2)
    X = rng.rand(n_samples, n_features)
    bad_degrees_of_freedom_prior_ = n_features - 1.0
    bgmm = BayesianGaussianMixture(degrees_of_freedom_prior=bad_degrees_of_freedom_prior_, random_state=rng)
    msg = f"The parameter 'degrees_of_freedom_prior' should be greater than {n_features - 1}, but got {bad_degrees_of_freedom_prior_:.3f}."
    with pytest.raises(ValueError, match=msg):
        bgmm.fit(X)
    degrees_of_freedom_prior = rng.rand() + n_features - 1.0
    bgmm = BayesianGaussianMixture(degrees_of_freedom_prior=degrees_of_freedom_prior, random_state=rng).fit(X)
    assert_almost_equal(degrees_of_freedom_prior, bgmm.degrees_of_freedom_prior_)
    degrees_of_freedom_prior_default = n_features
    bgmm = BayesianGaussianMixture(degrees_of_freedom_prior=degrees_of_freedom_prior_default, random_state=rng).fit(X)
    assert_almost_equal(degrees_of_freedom_prior_default, bgmm.degrees_of_freedom_prior_)
    covariance_prior = {'full': np.cov(X.T, bias=1) + 10, 'tied': np.cov(X.T, bias=1) + 5, 'diag': np.diag(np.atleast_2d(np.cov(X.T, bias=1))) + 3, 'spherical': rng.rand()}
    bgmm = BayesianGaussianMixture(random_state=rng)
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        bgmm.covariance_type = cov_type
        bgmm.covariance_prior = covariance_prior[cov_type]
        bgmm.fit(X)
        assert_almost_equal(covariance_prior[cov_type], bgmm.covariance_prior_)
    covariance_prior_default = {'full': np.atleast_2d(np.cov(X.T)), 'tied': np.atleast_2d(np.cov(X.T)), 'diag': np.var(X, axis=0, ddof=1), 'spherical': np.var(X, axis=0, ddof=1).mean()}
    bgmm = BayesianGaussianMixture(random_state=0)
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        bgmm.covariance_type = cov_type
        bgmm.fit(X)
        assert_almost_equal(covariance_prior_default[cov_type], bgmm.covariance_prior_)

def test_bayesian_mixture_check_is_fitted():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 2)
    bgmm = BayesianGaussianMixture(random_state=rng)
    X = rng.rand(n_samples, n_features)
    msg = 'This BayesianGaussianMixture instance is not fitted yet.'
    with pytest.raises(ValueError, match=msg):
        bgmm.score(X)

def test_bayesian_mixture_weights():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 2)
    X = rng.rand(n_samples, n_features)
    bgmm = BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_distribution', n_components=3, random_state=rng).fit(X)
    expected_weights = bgmm.weight_concentration_ / np.sum(bgmm.weight_concentration_)
    assert_almost_equal(expected_weights, bgmm.weights_)
    assert_almost_equal(np.sum(bgmm.weights_), 1.0)
    dpgmm = BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process', n_components=3, random_state=rng).fit(X)
    weight_dirichlet_sum = dpgmm.weight_concentration_[0] + dpgmm.weight_concentration_[1]
    tmp = dpgmm.weight_concentration_[1] / weight_dirichlet_sum
    expected_weights = dpgmm.weight_concentration_[0] / weight_dirichlet_sum * np.hstack((1, np.cumprod(tmp[:-1])))
    expected_weights /= np.sum(expected_weights)
    assert_almost_equal(expected_weights, dpgmm.weights_)
    assert_almost_equal(np.sum(dpgmm.weights_), 1.0)

@ignore_warnings(category=ConvergenceWarning)
def test_monotonic_likelihood():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=20)
    n_components = rand_data.n_components
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            X = rand_data.X[covar_type]
            bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type=covar_type, warm_start=True, max_iter=1, random_state=rng, tol=0.001)
            current_lower_bound = -np.inf
            for _ in range(600):
                prev_lower_bound = current_lower_bound
                current_lower_bound = bgmm.fit(X).lower_bound_
                assert current_lower_bound >= prev_lower_bound
                if bgmm.converged_:
                    break
            assert bgmm.converged_

def test_compare_covar_type():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    X = rand_data.X['full']
    n_components = rand_data.n_components
    for prior_type in PRIOR_TYPE:
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='full', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        full_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_[:, np.newaxis, np.newaxis]
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='tied', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        tied_covariance = bgmm.covariances_ * bgmm.degrees_of_freedom_
        assert_almost_equal(tied_covariance, np.mean(full_covariances, 0))
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='diag', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        diag_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_[:, np.newaxis]
        assert_almost_equal(diag_covariances, np.array([np.diag(cov) for cov in full_covariances]))
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='spherical', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        spherical_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_
        assert_almost_equal(spherical_covariances, np.mean(diag_covariances, 1))

@ignore_warnings(category=ConvergenceWarning)
def test_check_covariance_precision():
    if False:
        return 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    (n_components, n_features) = (2 * rand_data.n_components, 2)
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=100, random_state=rng, tol=0.001, reg_covar=0)
    for covar_type in COVARIANCE_TYPE:
        bgmm.covariance_type = covar_type
        bgmm.fit(rand_data.X[covar_type])
        if covar_type == 'full':
            for (covar, precision) in zip(bgmm.covariances_, bgmm.precisions_):
                assert_almost_equal(np.dot(covar, precision), np.eye(n_features))
        elif covar_type == 'tied':
            assert_almost_equal(np.dot(bgmm.covariances_, bgmm.precisions_), np.eye(n_features))
        elif covar_type == 'diag':
            assert_almost_equal(bgmm.covariances_ * bgmm.precisions_, np.ones((n_components, n_features)))
        else:
            assert_almost_equal(bgmm.covariances_ * bgmm.precisions_, np.ones(n_components))

@ignore_warnings(category=ConvergenceWarning)
def test_invariant_translation():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=100)
    n_components = 2 * rand_data.n_components
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            X = rand_data.X[covar_type]
            bgmm1 = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=n_components, max_iter=100, random_state=0, tol=0.001, reg_covar=0).fit(X)
            bgmm2 = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=n_components, max_iter=100, random_state=0, tol=0.001, reg_covar=0).fit(X + 100)
            assert_almost_equal(bgmm1.means_, bgmm2.means_ - 100)
            assert_almost_equal(bgmm1.weights_, bgmm2.weights_)
            assert_almost_equal(bgmm1.covariances_, bgmm2.covariances_)

@pytest.mark.filterwarnings('ignore:.*did not converge.*')
@pytest.mark.parametrize('seed, max_iter, tol', [(0, 2, 1e-07), (1, 2, 0.1), (3, 300, 1e-07), (4, 300, 0.1)])
def test_bayesian_mixture_fit_predict(seed, max_iter, tol):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(seed)
    rand_data = RandomData(rng, n_samples=50, scale=7)
    n_components = 2 * rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        bgmm1 = BayesianGaussianMixture(n_components=n_components, max_iter=max_iter, random_state=rng, tol=tol, reg_covar=0)
        bgmm1.covariance_type = covar_type
        bgmm2 = copy.deepcopy(bgmm1)
        X = rand_data.X[covar_type]
        Y_pred1 = bgmm1.fit(X).predict(X)
        Y_pred2 = bgmm2.fit_predict(X)
        assert_array_equal(Y_pred1, Y_pred2)

def test_bayesian_mixture_fit_predict_n_init():
    if False:
        i = 10
        return i + 15
    X = np.random.RandomState(0).randn(50, 5)
    gm = BayesianGaussianMixture(n_components=5, n_init=10, random_state=0)
    y_pred1 = gm.fit_predict(X)
    y_pred2 = gm.predict(X)
    assert_array_equal(y_pred1, y_pred2)

def test_bayesian_mixture_predict_predict_proba():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            X = rand_data.X[covar_type]
            Y = rand_data.Y
            bgmm = BayesianGaussianMixture(n_components=rand_data.n_components, random_state=rng, weight_concentration_prior_type=prior_type, covariance_type=covar_type)
            msg = "This BayesianGaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            with pytest.raises(NotFittedError, match=msg):
                bgmm.predict(X)
            bgmm.fit(X)
            Y_pred = bgmm.predict(X)
            Y_pred_proba = bgmm.predict_proba(X).argmax(axis=1)
            assert_array_equal(Y_pred, Y_pred_proba)
            assert adjusted_rand_score(Y, Y_pred) >= 0.95