"""Gaussian processes regression."""
import warnings
from numbers import Integral, Real
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
from ..preprocessing._data import _handle_zeros_in_scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from .kernels import RBF, Kernel
from .kernels import ConstantKernel as C
GPR_CHOLESKY_LOWER = True

class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of [RW2006]_.

    In addition to standard scikit-learn estimator API,
    :class:`GaussianProcessRegressor`:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method `sample_y(X)`, which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method `log_marginal_likelihood(theta)`, which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    To learn the difference between a point-estimate approach vs. a more
    Bayesian modelling approach, refer to the example entitled
    :ref:`sphx_glr_auto_examples_gaussian_process_plot_compare_gpr_krr.py`.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(1.0, length_scale_bounds="fixed")`` is used as default. Note that
        the kernel hyperparameters are optimized during fitting unless the
        bounds are marked as "fixed".

    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with :class:`~sklearn.linear_model.Ridge`.

    optimizer : "fmin_l_bfgs_b", callable or None, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func': the objective function to be minimized, which
                #   takes the hyperparameters theta as a parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the L-BFGS-B algorithm from `scipy.optimize.minimize`
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are: `{'fmin_l_bfgs_b'}`.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts_optimizer == 0` implies that one
        run is performed.

    normalize_y : bool, default=False
        Whether or not to normalize the target values `y` by removing the mean
        and scaling to unit-variance. This is recommended for cases where
        zero-mean, unit-variance priors are used. Note that, in this
        implementation, the normalisation is reversed before the GP predictions
        are reported.

        .. versionchanged:: 0.23

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    n_targets : int, default=None
        The number of dimensions of the target values. Used to decide the number
        of outputs when sampling from the prior distributions (i.e. calling
        :meth:`sample_y` before :meth:`fit`). This parameter is ignored once
        :meth:`fit` has been called.

        .. versionadded:: 1.3

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values in training data (also required for prediction).

    kernel_ : kernel instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``.

    alpha_ : array-like of shape (n_samples,)
        Dual coefficients of training data points in kernel space.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianProcessClassifier : Gaussian process classification (GPC)
        based on Laplace approximation.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))
    """
    _parameter_constraints: dict = {'kernel': [None, Kernel], 'alpha': [Interval(Real, 0, None, closed='left'), np.ndarray], 'optimizer': [StrOptions({'fmin_l_bfgs_b'}), callable, None], 'n_restarts_optimizer': [Interval(Integral, 0, None, closed='left')], 'normalize_y': ['boolean'], 'copy_X_train': ['boolean'], 'n_targets': [Interval(Integral, 1, None, closed='left'), None], 'random_state': ['random_state']}

    def __init__(self, kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
        if False:
            while True:
                i = 10
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        if False:
            while True:
                i = 10
        'Fit Gaussian process regression model.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Feature vectors or other representations of training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Target values.\n\n        Returns\n        -------\n        self : object\n            GaussianProcessRegressor class instance.\n        '
        if self.kernel is None:
            self.kernel_ = C(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
        else:
            self.kernel_ = clone(self.kernel)
        self._rng = check_random_state(self.random_state)
        if self.kernel_.requires_vector_input:
            (dtype, ensure_2d) = ('numeric', True)
        else:
            (dtype, ensure_2d) = (None, False)
        (X, y) = self._validate_data(X, y, multi_output=True, y_numeric=True, ensure_2d=ensure_2d, dtype=dtype)
        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
        if self.n_targets is not None and n_targets_seen != self.n_targets:
            raise ValueError(f'The number of targets seen in `y` is different from the parameter `n_targets`. Got {n_targets_seen} != {self.n_targets}.')
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)
        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(f'alpha must be a scalar or an array with same number of entries as y. ({self.alpha.shape[0]} != {y.shape[0]})')
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        if self.optimizer is not None and self.kernel_.n_dims > 0:

            def obj_func(theta, eval_gradient=True):
                if False:
                    while True:
                        i = 10
                if eval_gradient:
                    (lml, grad) = self.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
                    return (-lml, -grad)
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)
            optima = [self._constrained_optimization(obj_func, self.kernel_.theta, self.kernel_.bounds)]
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError('Multiple optimizer restarts (n_restarts_optimizer>0) requires that all bounds are finite.')
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(self._constrained_optimization(obj_func, theta_initial, bounds))
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(self.kernel_.theta, clone_kernel=False)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (f"The kernel, {self.kernel_}, is not returning a positive definite matrix. Try gradually increasing the 'alpha' parameter of your GaussianProcessRegressor estimator.",) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, GPR_CHOLESKY_LOWER), self.y_train_, check_finite=False)
        return self

    def predict(self, X, return_std=False, return_cov=False):
        if False:
            i = 10
            return i + 15
        'Predict using the Gaussian process regression model.\n\n        We can also predict based on an unfitted model by using the GP prior.\n        In addition to the mean of the predictive distribution, optionally also\n        returns its standard deviation (`return_std=True`) or covariance\n        (`return_cov=True`). Note that at most one of the two can be requested.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Query points where the GP is evaluated.\n\n        return_std : bool, default=False\n            If True, the standard-deviation of the predictive distribution at\n            the query points is returned along with the mean.\n\n        return_cov : bool, default=False\n            If True, the covariance of the joint predictive distribution at\n            the query points is returned along with the mean.\n\n        Returns\n        -------\n        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)\n            Mean of predictive distribution a query points.\n\n        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional\n            Standard deviation of predictive distribution at query points.\n            Only returned when `return_std` is True.\n\n        y_cov : ndarray of shape (n_samples, n_samples) or                 (n_samples, n_samples, n_targets), optional\n            Covariance of joint predictive distribution a query points.\n            Only returned when `return_cov` is True.\n        '
        if return_std and return_cov:
            raise RuntimeError('At most one of return_std or return_cov can be requested.')
        if self.kernel is None or self.kernel.requires_vector_input:
            (dtype, ensure_2d) = ('numeric', True)
        else:
            (dtype, ensure_2d) = (None, False)
        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        if not hasattr(self, 'X_train_'):
            if self.kernel is None:
                kernel = C(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
            else:
                kernel = self.kernel
            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()
            if return_cov:
                y_cov = kernel(X)
                if n_targets > 1:
                    y_cov = np.repeat(np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1)
                return (y_mean, y_cov)
            elif return_std:
                y_var = kernel.diag(X)
                if n_targets > 1:
                    y_var = np.repeat(np.expand_dims(y_var, -1), repeats=n_targets, axis=-1)
                return (y_mean, np.sqrt(y_var))
            else:
                return y_mean
        else:
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.alpha_
            y_mean = self._y_train_std * y_mean + self._y_train_mean
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)
            V = solve_triangular(self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False)
            if return_cov:
                y_cov = self.kernel_(X) - V.T @ V
                y_cov = np.outer(y_cov, self._y_train_std ** 2).reshape(*y_cov.shape, -1)
                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)
                return (y_mean, y_cov)
            elif return_std:
                y_var = self.kernel_.diag(X).copy()
                y_var -= np.einsum('ij,ji->i', V.T, V)
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn('Predicted variances smaller than 0. Setting those variances to 0.')
                    y_var[y_var_negative] = 0.0
                y_var = np.outer(y_var, self._y_train_std ** 2).reshape(*y_var.shape, -1)
                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)
                return (y_mean, np.sqrt(y_var))
            else:
                return y_mean

    def sample_y(self, X, n_samples=1, random_state=0):
        if False:
            for i in range(10):
                print('nop')
        'Draw samples from Gaussian process and evaluate at X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples_X, n_features) or list of object\n            Query points where the GP is evaluated.\n\n        n_samples : int, default=1\n            Number of samples drawn from the Gaussian process per query point.\n\n        random_state : int, RandomState instance or None, default=0\n            Determines random number generation to randomly draw samples.\n            Pass an int for reproducible results across multiple function\n            calls.\n            See :term:`Glossary <random_state>`.\n\n        Returns\n        -------\n        y_samples : ndarray of shape (n_samples_X, n_samples), or             (n_samples_X, n_targets, n_samples)\n            Values of n_samples samples drawn from Gaussian process and\n            evaluated at query points.\n        '
        rng = check_random_state(random_state)
        (y_mean, y_cov) = self.predict(X, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [rng.multivariate_normal(y_mean[:, target], y_cov[..., target], n_samples).T[:, np.newaxis] for target in range(y_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        if False:
            i = 10
            return i + 15
        'Return log-marginal likelihood of theta for training data.\n\n        Parameters\n        ----------\n        theta : array-like of shape (n_kernel_params,) default=None\n            Kernel hyperparameters for which the log-marginal likelihood is\n            evaluated. If None, the precomputed log_marginal_likelihood\n            of ``self.kernel_.theta`` is returned.\n\n        eval_gradient : bool, default=False\n            If True, the gradient of the log-marginal likelihood with respect\n            to the kernel hyperparameters at position theta is returned\n            additionally. If True, theta must not be None.\n\n        clone_kernel : bool, default=True\n            If True, the kernel attribute is copied. If False, the kernel\n            attribute is modified, but may result in a performance improvement.\n\n        Returns\n        -------\n        log_likelihood : float\n            Log-marginal likelihood of theta for training data.\n\n        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional\n            Gradient of the log-marginal likelihood with respect to the kernel\n            hyperparameters at position theta.\n            Only returned when eval_gradient is True.\n        '
        if theta is None:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated for theta!=None')
            return self.log_marginal_likelihood_value_
        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta
        if eval_gradient:
            (K, K_gradient) = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)
        log_likelihood_dims = -0.5 * np.einsum('ik,ik->k', y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)
        if eval_gradient:
            inner_term = np.einsum('ik,jk->ijk', alpha, alpha)
            K_inv = cho_solve((L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False)
            inner_term -= K_inv[..., np.newaxis]
            log_likelihood_gradient_dims = 0.5 * np.einsum('ijl,jik->kl', inner_term, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
        if eval_gradient:
            return (log_likelihood, log_likelihood_gradient)
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if False:
            i = 10
            return i + 15
        if self.optimizer == 'fmin_l_bfgs_b':
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B', jac=True, bounds=bounds)
            _check_optimize_result('lbfgs', opt_res)
            (theta_opt, func_min) = (opt_res.x, opt_res.fun)
        elif callable(self.optimizer):
            (theta_opt, func_min) = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f'Unknown optimizer {self.optimizer}.')
        return (theta_opt, func_min)

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'requires_fit': False}