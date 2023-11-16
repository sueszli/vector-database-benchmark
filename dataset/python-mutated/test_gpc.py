"""Testing for Gaussian process classification """
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, CompoundKernel, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_almost_equal, assert_array_equal

def f(x):
    if False:
        print('Hello World!')
    return np.sin(x)
X = np.atleast_2d(np.linspace(0, 10, 30)).T
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T
y = np.array(f(X).ravel() > 0, dtype=int)
fX = f(X).ravel()
y_mc = np.empty(y.shape, dtype=int)
y_mc[fX < -0.35] = 0
y_mc[(fX >= -0.35) & (fX < 0.35)] = 1
y_mc[fX > 0.35] = 2
fixed_kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')
kernels = [RBF(length_scale=0.1), fixed_kernel, RBF(length_scale=1.0, length_scale_bounds=(0.001, 1000.0)), C(1.0, (0.01, 100.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.001, 1000.0))]
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]

@pytest.mark.parametrize('kernel', kernels)
def test_predict_consistent(kernel):
    if False:
        print('Hello World!')
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)

def test_predict_consistent_structured():
    if False:
        print('Hello World!')
    X = ['A', 'AB', 'B']
    y = np.array([True, False, True])
    kernel = MiniSeqKernel(baseline_similarity_bounds='fixed')
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_lml_improving(kernel):
    if False:
        return 10
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) > gpc.log_marginal_likelihood(kernel.theta)

@pytest.mark.parametrize('kernel', kernels)
def test_lml_precomputed(kernel):
    if False:
        return 10
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_almost_equal(gpc.log_marginal_likelihood(gpc.kernel_.theta), gpc.log_marginal_likelihood(), 7)

@pytest.mark.parametrize('kernel', kernels)
def test_lml_without_cloning_kernel(kernel):
    if False:
        for i in range(10):
            print('nop')
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpc.kernel_.theta.shape, dtype=np.float64)
    gpc.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpc.kernel_.theta, input_theta, 7)

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    if False:
        while True:
            i = 10
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    (lml, lml_gradient) = gpc.log_marginal_likelihood(gpc.kernel_.theta, True)
    assert np.all((np.abs(lml_gradient) < 0.0001) | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 0]) | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 1]))

@pytest.mark.parametrize('kernel', kernels)
def test_lml_gradient(kernel):
    if False:
        i = 10
        return i + 15
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    (lml, lml_gradient) = gpc.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = approx_fprime(kernel.theta, lambda theta: gpc.log_marginal_likelihood(theta, False), 1e-10)
    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)

def test_random_starts(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    (n_samples, n_features) = (25, 2)
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1) > 0
    kernel = C(1.0, (0.01, 100.0)) * RBF(length_scale=[0.001] * n_features, length_scale_bounds=[(0.0001, 100.0)] * n_features)
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=global_random_seed).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_custom_optimizer(kernel, global_random_seed):
    if False:
        for i in range(10):
            print('nop')

    def optimizer(obj_func, initial_theta, bounds):
        if False:
            while True:
                i = 10
        rng = np.random.RandomState(global_random_seed)
        (theta_opt, func_min) = (initial_theta, obj_func(initial_theta, eval_gradient=False))
        for _ in range(10):
            theta = np.atleast_1d(rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1])))
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                (theta_opt, func_min) = (theta, f)
        return (theta_opt, func_min)
    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer)
    gpc.fit(X, y_mc)
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) >= gpc.log_marginal_likelihood(kernel.theta)

@pytest.mark.parametrize('kernel', kernels)
def test_multi_class(kernel):
    if False:
        for i in range(10):
            print('nop')
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)
    y_prob = gpc.predict_proba(X2)
    assert_almost_equal(y_prob.sum(1), 1)
    y_pred = gpc.predict(X2)
    assert_array_equal(np.argmax(y_prob, 1), y_pred)

@pytest.mark.parametrize('kernel', kernels)
def test_multi_class_n_jobs(kernel):
    if False:
        print('Hello World!')
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)
    gpc_2 = GaussianProcessClassifier(kernel=kernel, n_jobs=2)
    gpc_2.fit(X, y_mc)
    y_prob = gpc.predict_proba(X2)
    y_prob_2 = gpc_2.predict_proba(X2)
    assert_almost_equal(y_prob, y_prob_2)

def test_warning_bounds():
    if False:
        print('Hello World!')
    kernel = RBF(length_scale_bounds=[1e-05, 0.001])
    gpc = GaussianProcessClassifier(kernel=kernel)
    warning_message = 'The optimal value found for dimension 0 of parameter length_scale is close to the specified upper bound 0.001. Increasing the bound and calling fit again may find a better value.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpc.fit(X, y)
    kernel_sum = WhiteKernel(noise_level_bounds=[1e-05, 0.001]) + RBF(length_scale_bounds=[1000.0, 100000.0])
    gpc_sum = GaussianProcessClassifier(kernel=kernel_sum)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gpc_sum.fit(X, y)
        assert len(record) == 2
        assert issubclass(record[0].category, ConvergenceWarning)
        assert record[0].message.args[0] == 'The optimal value found for dimension 0 of parameter k1__noise_level is close to the specified upper bound 0.001. Increasing the bound and calling fit again may find a better value.'
        assert issubclass(record[1].category, ConvergenceWarning)
        assert record[1].message.args[0] == 'The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound 1000.0. Decreasing the bound and calling fit again may find a better value.'
    X_tile = np.tile(X, 2)
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[10.0, 100.0])
    gpc_dims = GaussianProcessClassifier(kernel=kernel_dims)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gpc_dims.fit(X_tile, y)
        assert len(record) == 2
        assert issubclass(record[0].category, ConvergenceWarning)
        assert record[0].message.args[0] == 'The optimal value found for dimension 0 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.'
        assert issubclass(record[1].category, ConvergenceWarning)
        assert record[1].message.args[0] == 'The optimal value found for dimension 1 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.'

@pytest.mark.parametrize('params, error_type, err_msg', [({'kernel': CompoundKernel(0)}, ValueError, 'kernel cannot be a CompoundKernel')])
def test_gpc_fit_error(params, error_type, err_msg):
    if False:
        i = 10
        return i + 15
    'Check that expected error are raised during fit.'
    gpc = GaussianProcessClassifier(**params)
    with pytest.raises(error_type, match=err_msg):
        gpc.fit(X, y)