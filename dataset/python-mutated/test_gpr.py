"""Testing for Gaussian process regression """
import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_less

def f(x):
    if False:
        print('Hello World!')
    return x * np.sin(x)
X = np.atleast_2d([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).T
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T
y = f(X).ravel()
fixed_kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')
kernels = [RBF(length_scale=1.0), fixed_kernel, RBF(length_scale=1.0, length_scale_bounds=(0.001, 1000.0)), C(1.0, (0.01, 100.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.001, 1000.0)), C(1.0, (0.01, 100.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.001, 1000.0)) + C(1e-05, (1e-05, 100.0)), C(0.1, (0.01, 100.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.001, 1000.0)) + C(1e-05, (1e-05, 100.0))]
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]

@pytest.mark.parametrize('kernel', kernels)
def test_gpr_interpolation(kernel):
    if False:
        for i in range(10):
            print('nop')
    if sys.maxsize <= 2 ** 32:
        pytest.xfail('This test may fail on 32 bit Python')
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    (y_pred, y_cov) = gpr.predict(X, return_cov=True)
    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.0)

def test_gpr_interpolation_structured():
    if False:
        for i in range(10):
            print('nop')
    kernel = MiniSeqKernel(baseline_similarity_bounds='fixed')
    X = ['A', 'B', 'C']
    y = np.array([1, 2, 3])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    (y_pred, y_cov) = gpr.predict(X, return_cov=True)
    assert_almost_equal(kernel(X, eval_gradient=True)[1].ravel(), (1 - np.eye(len(X))).ravel())
    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.0)

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_lml_improving(kernel):
    if False:
        while True:
            i = 10
    if sys.maxsize <= 2 ** 32:
        pytest.xfail('This test may fail on 32 bit Python')
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(kernel.theta)

@pytest.mark.parametrize('kernel', kernels)
def test_lml_precomputed(kernel):
    if False:
        return 10
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) == pytest.approx(gpr.log_marginal_likelihood())

@pytest.mark.parametrize('kernel', kernels)
def test_lml_without_cloning_kernel(kernel):
    if False:
        i = 10
        return i + 15
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpr.kernel_.theta.shape, dtype=np.float64)
    gpr.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpr.kernel_.theta, input_theta, 7)

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    if False:
        return 10
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    (lml, lml_gradient) = gpr.log_marginal_likelihood(gpr.kernel_.theta, True)
    assert np.all((np.abs(lml_gradient) < 0.0001) | (gpr.kernel_.theta == gpr.kernel_.bounds[:, 0]) | (gpr.kernel_.theta == gpr.kernel_.bounds[:, 1]))

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_solution_inside_bounds(kernel):
    if False:
        i = 10
        return i + 15
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    bounds = gpr.kernel_.bounds
    max_ = np.finfo(gpr.kernel_.theta.dtype).max
    tiny = 1e-10
    bounds[~np.isfinite(bounds[:, 1]), 1] = max_
    assert_array_less(bounds[:, 0], gpr.kernel_.theta + tiny)
    assert_array_less(gpr.kernel_.theta, bounds[:, 1] + tiny)

@pytest.mark.parametrize('kernel', kernels)
def test_lml_gradient(kernel):
    if False:
        return 10
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    (lml, lml_gradient) = gpr.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = approx_fprime(kernel.theta, lambda theta: gpr.log_marginal_likelihood(theta, False), 1e-10)
    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)

@pytest.mark.parametrize('kernel', kernels)
def test_prior(kernel):
    if False:
        i = 10
        return i + 15
    gpr = GaussianProcessRegressor(kernel=kernel)
    (y_mean, y_cov) = gpr.predict(X, return_cov=True)
    assert_almost_equal(y_mean, 0, 5)
    if len(gpr.kernel.theta) > 1:
        assert_almost_equal(np.diag(y_cov), np.exp(kernel.theta[0]), 5)
    else:
        assert_almost_equal(np.diag(y_cov), 1, 5)

@pytest.mark.parametrize('kernel', kernels)
def test_sample_statistics(kernel):
    if False:
        return 10
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    (y_mean, y_cov) = gpr.predict(X2, return_cov=True)
    samples = gpr.sample_y(X2, 300000)
    assert_almost_equal(y_mean, np.mean(samples, 1), 1)
    assert_almost_equal(np.diag(y_cov) / np.diag(y_cov).max(), np.var(samples, 1) / np.diag(y_cov).max(), 1)

def test_no_optimizer():
    if False:
        i = 10
        return i + 15
    kernel = RBF(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(X, y)
    assert np.exp(gpr.kernel_.theta) == 1.0

@pytest.mark.parametrize('kernel', kernels)
@pytest.mark.parametrize('target', [y, np.ones(X.shape[0], dtype=np.float64)])
def test_predict_cov_vs_std(kernel, target):
    if False:
        i = 10
        return i + 15
    if sys.maxsize <= 2 ** 32:
        pytest.xfail('This test may fail on 32 bit Python')
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    (y_mean, y_cov) = gpr.predict(X2, return_cov=True)
    (y_mean, y_std) = gpr.predict(X2, return_std=True)
    assert_almost_equal(np.sqrt(np.diag(y_cov)), y_std)

def test_anisotropic_kernel():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (50, 2))
    y = X[:, 0] + 0.1 * X[:, 1]
    kernel = RBF([1.0, 1.0])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert np.exp(gpr.kernel_.theta[1]) > np.exp(gpr.kernel_.theta[0]) * 5

def test_random_starts():
    if False:
        while True:
            i = 10
    (n_samples, n_features) = (25, 2)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1) + rng.normal(scale=0.1, size=n_samples)
    kernel = C(1.0, (0.01, 100.0)) * RBF(length_scale=[1.0] * n_features, length_scale_bounds=[(0.0001, 100.0)] * n_features) + WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-05, 10.0))
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=0).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml

@pytest.mark.parametrize('kernel', kernels)
def test_y_normalization(kernel):
    if False:
        while True:
            i = 10
    "\n    Test normalization of the target values in GP\n\n    Fitting non-normalizing GP on normalized y and fitting normalizing GP\n    on unnormalized y should yield identical results. Note that, here,\n    'normalized y' refers to y that has been made zero mean and unit\n    variance.\n\n    "
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y_norm)
    gpr_norm = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_norm.fit(X, y)
    (y_pred, y_pred_std) = gpr.predict(X2, return_std=True)
    y_pred = y_pred * y_std + y_mean
    y_pred_std = y_pred_std * y_std
    (y_pred_norm, y_pred_std_norm) = gpr_norm.predict(X2, return_std=True)
    assert_almost_equal(y_pred, y_pred_norm)
    assert_almost_equal(y_pred_std, y_pred_std_norm)
    (_, y_cov) = gpr.predict(X2, return_cov=True)
    y_cov = y_cov * y_std ** 2
    (_, y_cov_norm) = gpr_norm.predict(X2, return_cov=True)
    assert_almost_equal(y_cov, y_cov_norm)

def test_large_variance_y():
    if False:
        while True:
            i = 10
    "\n    Here we test that, when noramlize_y=True, our GP can produce a\n    sensible fit to training data whose variance is significantly\n    larger than unity. This test was made in response to issue #15612.\n\n    GP predictions are verified against predictions that were made\n    using GPy which, here, is treated as the 'gold standard'. Note that we\n    only investigate the RBF kernel here, as that is what was used in the\n    GPy implementation.\n\n    The following code can be used to recreate the GPy data:\n\n    --------------------------------------------------------------------------\n    import GPy\n\n    kernel_gpy = GPy.kern.RBF(input_dim=1, lengthscale=1.)\n    gpy = GPy.models.GPRegression(X, np.vstack(y_large), kernel_gpy)\n    gpy.optimize()\n    y_pred_gpy, y_var_gpy = gpy.predict(X2)\n    y_pred_std_gpy = np.sqrt(y_var_gpy)\n    --------------------------------------------------------------------------\n    "
    y_large = 10 * y
    RBF_params = {'length_scale': 1.0}
    kernel = RBF(**RBF_params)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_large)
    (y_pred, y_pred_std) = gpr.predict(X2, return_std=True)
    y_pred_gpy = np.array([15.16918303, -27.98707845, -39.31636019, 14.52605515, 69.18503589])
    y_pred_std_gpy = np.array([7.78860962, 3.83179178, 0.63149951, 0.52745188, 0.86170042])
    assert_allclose(y_pred, y_pred_gpy, rtol=0.07, atol=0)
    assert_allclose(y_pred_std, y_pred_std_gpy, rtol=0.15, atol=0)

def test_y_multioutput():
    if False:
        print('Hello World!')
    y_2d = np.vstack((y, y * 2)).T
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gpr.fit(X, y)
    gpr_2d = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gpr_2d.fit(X, y_2d)
    (y_pred_1d, y_std_1d) = gpr.predict(X2, return_std=True)
    (y_pred_2d, y_std_2d) = gpr_2d.predict(X2, return_std=True)
    (_, y_cov_1d) = gpr.predict(X2, return_cov=True)
    (_, y_cov_2d) = gpr_2d.predict(X2, return_cov=True)
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 0])
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 1] / 2)
    for target in range(y_2d.shape[1]):
        assert_almost_equal(y_std_1d, y_std_2d[..., target])
        assert_almost_equal(y_cov_1d, y_cov_2d[..., target])
    y_sample_1d = gpr.sample_y(X2, n_samples=10)
    y_sample_2d = gpr_2d.sample_y(X2, n_samples=10)
    assert y_sample_1d.shape == (5, 10)
    assert y_sample_2d.shape == (5, 2, 10)
    assert_almost_equal(y_sample_1d, y_sample_2d[:, 0, :])
    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X, y)
        gpr_2d = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr_2d.fit(X, np.vstack((y, y)).T)
        assert_almost_equal(gpr.kernel_.theta, gpr_2d.kernel_.theta, 4)

@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_custom_optimizer(kernel):
    if False:
        return 10

    def optimizer(obj_func, initial_theta, bounds):
        if False:
            return 10
        rng = np.random.RandomState(0)
        (theta_opt, func_min) = (initial_theta, obj_func(initial_theta, eval_gradient=False))
        for _ in range(50):
            theta = np.atleast_1d(rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1])))
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                (theta_opt, func_min) = (theta, f)
        return (theta_opt, func_min)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer)
    gpr.fit(X, y)
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(gpr.kernel.theta)

def test_gpr_correct_error_message():
    if False:
        for i in range(10):
            print('nop')
    X = np.arange(12).reshape(6, -1)
    y = np.ones(6)
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    message = "The kernel, %s, is not returning a positive definite matrix. Try gradually increasing the 'alpha' parameter of your GaussianProcessRegressor estimator." % kernel
    with pytest.raises(np.linalg.LinAlgError, match=re.escape(message)):
        gpr.fit(X, y)

@pytest.mark.parametrize('kernel', kernels)
def test_duplicate_input(kernel):
    if False:
        while True:
            i = 10
    gpr_equal_inputs = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    gpr_similar_inputs = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    X_ = np.vstack((X, X[0]))
    y_ = np.hstack((y, y[0] + 1))
    gpr_equal_inputs.fit(X_, y_)
    X_ = np.vstack((X, X[0] + 1e-15))
    y_ = np.hstack((y, y[0] + 1))
    gpr_similar_inputs.fit(X_, y_)
    X_test = np.linspace(0, 10, 100)[:, None]
    (y_pred_equal, y_std_equal) = gpr_equal_inputs.predict(X_test, return_std=True)
    (y_pred_similar, y_std_similar) = gpr_similar_inputs.predict(X_test, return_std=True)
    assert_almost_equal(y_pred_equal, y_pred_similar)
    assert_almost_equal(y_std_equal, y_std_similar)

def test_no_fit_default_predict():
    if False:
        return 10
    default_kernel = C(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
    gpr1 = GaussianProcessRegressor()
    (_, y_std1) = gpr1.predict(X, return_std=True)
    (_, y_cov1) = gpr1.predict(X, return_cov=True)
    gpr2 = GaussianProcessRegressor(kernel=default_kernel)
    (_, y_std2) = gpr2.predict(X, return_std=True)
    (_, y_cov2) = gpr2.predict(X, return_cov=True)
    assert_array_almost_equal(y_std1, y_std2)
    assert_array_almost_equal(y_cov1, y_cov2)

def test_warning_bounds():
    if False:
        print('Hello World!')
    kernel = RBF(length_scale_bounds=[1e-05, 0.001])
    gpr = GaussianProcessRegressor(kernel=kernel)
    warning_message = 'The optimal value found for dimension 0 of parameter length_scale is close to the specified upper bound 0.001. Increasing the bound and calling fit again may find a better value.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpr.fit(X, y)
    kernel_sum = WhiteKernel(noise_level_bounds=[1e-05, 0.001]) + RBF(length_scale_bounds=[1000.0, 100000.0])
    gpr_sum = GaussianProcessRegressor(kernel=kernel_sum)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gpr_sum.fit(X, y)
        assert len(record) == 2
        assert issubclass(record[0].category, ConvergenceWarning)
        assert record[0].message.args[0] == 'The optimal value found for dimension 0 of parameter k1__noise_level is close to the specified upper bound 0.001. Increasing the bound and calling fit again may find a better value.'
        assert issubclass(record[1].category, ConvergenceWarning)
        assert record[1].message.args[0] == 'The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound 1000.0. Decreasing the bound and calling fit again may find a better value.'
    X_tile = np.tile(X, 2)
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[10.0, 100.0])
    gpr_dims = GaussianProcessRegressor(kernel=kernel_dims)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gpr_dims.fit(X_tile, y)
        assert len(record) == 2
        assert issubclass(record[0].category, ConvergenceWarning)
        assert record[0].message.args[0] == 'The optimal value found for dimension 0 of parameter length_scale is close to the specified lower bound 10.0. Decreasing the bound and calling fit again may find a better value.'
        assert issubclass(record[1].category, ConvergenceWarning)
        assert record[1].message.args[0] == 'The optimal value found for dimension 1 of parameter length_scale is close to the specified lower bound 10.0. Decreasing the bound and calling fit again may find a better value.'

def test_bound_check_fixed_hyperparameter():
    if False:
        print('Hello World!')
    k1 = 50.0 ** 2 * RBF(length_scale=50.0)
    k2 = ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    kernel = k1 + k2
    GaussianProcessRegressor(kernel=kernel).fit(X, y)

@pytest.mark.parametrize('kernel', kernels)
def test_constant_target(kernel):
    if False:
        print('Hello World!')
    'Check that the std. dev. is affected to 1 when normalizing a constant\n    feature.\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/18318\n    NaN where affected to the target when scaling due to null std. dev. with\n    constant target.\n    '
    y_constant = np.ones(X.shape[0], dtype=np.float64)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_constant)
    assert gpr._y_train_std == pytest.approx(1.0)
    (y_pred, y_cov) = gpr.predict(X, return_cov=True)
    assert_allclose(y_pred, y_constant)
    assert_allclose(np.diag(y_cov), 0.0, atol=1e-09)
    (n_samples, n_targets) = (X.shape[0], 2)
    rng = np.random.RandomState(0)
    y = np.concatenate([rng.normal(size=(n_samples, 1)), np.full(shape=(n_samples, 1), fill_value=2)], axis=1)
    gpr.fit(X, y)
    (Y_pred, Y_cov) = gpr.predict(X, return_cov=True)
    assert_allclose(Y_pred[:, 1], 2)
    assert_allclose(np.diag(Y_cov[..., 1]), 0.0, atol=1e-09)
    assert Y_pred.shape == (n_samples, n_targets)
    assert Y_cov.shape == (n_samples, n_samples, n_targets)

def test_gpr_consistency_std_cov_non_invertible_kernel():
    if False:
        for i in range(10):
            print('nop')
    'Check the consistency between the returned std. dev. and the covariance.\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/19936\n    Inconsistencies were observed when the kernel cannot be inverted (or\n    numerically stable).\n    '
    kernel = C(898576.054, (1e-12, 1000000000000.0)) * RBF([591.32652, 1325.84051], (1e-12, 1000000000000.0)) + WhiteKernel(noise_level=1e-05)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None)
    X_train = np.array([[0.0, 0.0], [1.54919334, -0.77459667], [-1.54919334, 0.0], [0.0, -1.54919334], [0.77459667, 0.77459667], [-0.77459667, 1.54919334]])
    y_train = np.array([[-2.14882017e-10], [-4.66975823], [4.01823986], [-1.30303674], [-1.35760156], [3.31215668]])
    gpr.fit(X_train, y_train)
    X_test = np.array([[-1.93649167, -1.93649167], [1.93649167, -1.93649167], [-1.93649167, 1.93649167], [1.93649167, 1.93649167]])
    (pred1, std) = gpr.predict(X_test, return_std=True)
    (pred2, cov) = gpr.predict(X_test, return_cov=True)
    assert_allclose(std, np.sqrt(np.diagonal(cov)), rtol=1e-05)

@pytest.mark.parametrize('params, TypeError, err_msg', [({'alpha': np.zeros(100)}, ValueError, 'alpha must be a scalar or an array with same number of entries as y'), ({'kernel': WhiteKernel(noise_level_bounds=(-np.inf, np.inf)), 'n_restarts_optimizer': 2}, ValueError, 'requires that all bounds are finite')])
def test_gpr_fit_error(params, TypeError, err_msg):
    if False:
        print('Hello World!')
    'Check that expected error are raised during fit.'
    gpr = GaussianProcessRegressor(**params)
    with pytest.raises(TypeError, match=err_msg):
        gpr.fit(X, y)

def test_gpr_lml_error():
    if False:
        i = 10
        return i + 15
    'Check that we raise the proper error in the LML method.'
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)
    err_msg = 'Gradient can only be evaluated for theta!=None'
    with pytest.raises(ValueError, match=err_msg):
        gpr.log_marginal_likelihood(eval_gradient=True)

def test_gpr_predict_error():
    if False:
        for i in range(10):
            print('nop')
    'Check that we raise the proper error during predict.'
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)
    err_msg = 'At most one of return_std or return_cov can be requested.'
    with pytest.raises(RuntimeError, match=err_msg):
        gpr.predict(X, return_cov=True, return_std=True)

@pytest.mark.parametrize('normalize_y', [True, False])
@pytest.mark.parametrize('n_targets', [None, 1, 10])
def test_predict_shapes(normalize_y, n_targets):
    if False:
        i = 10
        return i + 15
    'Check the shapes of y_mean, y_std, and y_cov in single-output\n    (n_targets=None) and multi-output settings, including the edge case when\n    n_targets=1, where the sklearn convention is to squeeze the predictions.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/17394\n    https://github.com/scikit-learn/scikit-learn/issues/18065\n    https://github.com/scikit-learn/scikit-learn/issues/22174\n    '
    rng = np.random.RandomState(1234)
    (n_features, n_samples_train, n_samples_test) = (6, 9, 7)
    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)
    y_test_shape = (n_samples_test,)
    if n_targets is not None and n_targets > 1:
        y_test_shape = y_test_shape + (n_targets,)
    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_test, n_features)
    y_train = rng.randn(*y_train_shape)
    model = GaussianProcessRegressor(normalize_y=normalize_y)
    model.fit(X_train, y_train)
    (y_pred, y_std) = model.predict(X_test, return_std=True)
    (_, y_cov) = model.predict(X_test, return_cov=True)
    assert y_pred.shape == y_test_shape
    assert y_std.shape == y_test_shape
    assert y_cov.shape == (n_samples_test,) + y_test_shape

@pytest.mark.parametrize('normalize_y', [True, False])
@pytest.mark.parametrize('n_targets', [None, 1, 10])
def test_sample_y_shapes(normalize_y, n_targets):
    if False:
        while True:
            i = 10
    'Check the shapes of y_samples in single-output (n_targets=0) and\n    multi-output settings, including the edge case when n_targets=1, where the\n    sklearn convention is to squeeze the predictions.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/22175\n    '
    rng = np.random.RandomState(1234)
    (n_features, n_samples_train) = (6, 9)
    n_samples_X_test = 7
    n_samples_y_test = 5
    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)
    if n_targets is not None and n_targets > 1:
        y_test_shape = (n_samples_X_test, n_targets, n_samples_y_test)
    else:
        y_test_shape = (n_samples_X_test, n_samples_y_test)
    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_X_test, n_features)
    y_train = rng.randn(*y_train_shape)
    model = GaussianProcessRegressor(normalize_y=normalize_y)
    model.fit(X_train, y_train)
    y_samples = model.sample_y(X_test, n_samples=n_samples_y_test)
    assert y_samples.shape == y_test_shape

@pytest.mark.parametrize('n_targets', [None, 1, 2, 3])
@pytest.mark.parametrize('n_samples', [1, 5])
def test_sample_y_shape_with_prior(n_targets, n_samples):
    if False:
        i = 10
        return i + 15
    'Check the output shape of `sample_y` is consistent before and after `fit`.'
    rng = np.random.RandomState(1024)
    X = rng.randn(10, 3)
    y = rng.randn(10, n_targets if n_targets is not None else 1)
    model = GaussianProcessRegressor(n_targets=n_targets)
    shape_before_fit = model.sample_y(X, n_samples=n_samples).shape
    model.fit(X, y)
    shape_after_fit = model.sample_y(X, n_samples=n_samples).shape
    assert shape_before_fit == shape_after_fit

@pytest.mark.parametrize('n_targets', [None, 1, 2, 3])
def test_predict_shape_with_prior(n_targets):
    if False:
        i = 10
        return i + 15
    'Check the output shape of `predict` with prior distribution.'
    rng = np.random.RandomState(1024)
    n_sample = 10
    X = rng.randn(n_sample, 3)
    y = rng.randn(n_sample, n_targets if n_targets is not None else 1)
    model = GaussianProcessRegressor(n_targets=n_targets)
    (mean_prior, cov_prior) = model.predict(X, return_cov=True)
    (_, std_prior) = model.predict(X, return_std=True)
    model.fit(X, y)
    (mean_post, cov_post) = model.predict(X, return_cov=True)
    (_, std_post) = model.predict(X, return_std=True)
    assert mean_prior.shape == mean_post.shape
    assert cov_prior.shape == cov_post.shape
    assert std_prior.shape == std_post.shape

def test_n_targets_error():
    if False:
        for i in range(10):
            print('nop')
    'Check that an error is raised when the number of targets seen at fit is\n    inconsistent with n_targets.\n    '
    rng = np.random.RandomState(0)
    X = rng.randn(10, 3)
    y = rng.randn(10, 2)
    model = GaussianProcessRegressor(n_targets=1)
    with pytest.raises(ValueError, match='The number of targets seen in `y`'):
        model.fit(X, y)

class CustomKernel(C):
    """
    A custom kernel that has a diag method that returns the first column of the
    input matrix X. This is a helper for the test to check that the input
    matrix X is not mutated.
    """

    def diag(self, X):
        if False:
            for i in range(10):
                print('nop')
        return X[:, 0]

def test_gpr_predict_input_not_modified():
    if False:
        return 10
    '\n    Check that the input X is not modified by the predict method of the\n    GaussianProcessRegressor when setting return_std=True.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/24340\n    '
    gpr = GaussianProcessRegressor(kernel=CustomKernel()).fit(X, y)
    X2_copy = np.copy(X2)
    (_, _) = gpr.predict(X2, return_std=True)
    assert_allclose(X2, X2_copy)