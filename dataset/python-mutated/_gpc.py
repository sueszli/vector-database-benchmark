"""Gaussian processes classification."""
from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit
from ..base import BaseEstimator, ClassifierMixin, _fit_context, clone
from ..multiclass import OneVsOneClassifier, OneVsRestClassifier
from ..preprocessing import LabelEncoder
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from .kernels import RBF, CompoundKernel, Kernel
from .kernels import ConstantKernel as C
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654])[:, np.newaxis]

class _BinaryGaussianProcessClassifierLaplace(BaseEstimator):
    """Binary Gaussian process classification based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    optimizer : 'fmin_l_bfgs_b' or callable, default='fmin_l_bfgs_b'
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'L-BFGS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.

    max_iter_predict : int, default=100
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    warm_start : bool, default=False
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization. See :term:`the Glossary
        <warm_start>`.

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,)
        Target values in training data (also required for prediction)

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    kernel_ : kernl instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in X_train_

    pi_ : array-like of shape (n_samples,)
        The probabilities of the positive class for the training points
        X_train_

    W_sr_ : array-like of shape (n_samples,)
        Square root of W, the Hessian of log-likelihood of the latent function
        values for the observed labels. Since W is diagonal, only the diagonal
        of sqrt(W) is stored.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    """

    def __init__(self, kernel=None, *, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None):
        if False:
            print('Hello World!')
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X, y):
        if False:
            while True:
                i = 10
        'Fit Gaussian process classification model.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Feature vectors or other representations of training data.\n\n        y : array-like of shape (n_samples,)\n            Target values, must be binary.\n\n        Returns\n        -------\n        self : returns an instance of self.\n        '
        if self.kernel is None:
            self.kernel_ = C(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
        else:
            self.kernel_ = clone(self.kernel)
        self.rng = check_random_state(self.random_state)
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        label_encoder = LabelEncoder()
        self.y_train_ = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        if self.classes_.size > 2:
            raise ValueError('%s supports only binary classification. y contains classes %s' % (self.__class__.__name__, self.classes_))
        elif self.classes_.size == 1:
            raise ValueError('{0:s} requires 2 classes; got {1:d} class'.format(self.__class__.__name__, self.classes_.size))
        if self.optimizer is not None and self.kernel_.n_dims > 0:

            def obj_func(theta, eval_gradient=True):
                if False:
                    i = 10
                    return i + 15
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
                    theta_initial = np.exp(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
                    optima.append(self._constrained_optimization(obj_func, theta_initial, bounds))
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(self.kernel_.theta)
        K = self.kernel_(self.X_train_)
        (_, (self.pi_, self.W_sr_, self.L_, _, _)) = self._posterior_mode(K, return_temporaries=True)
        return self

    def predict(self, X):
        if False:
            return 10
        'Perform classification on an array of test vectors X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Query points where the GP is evaluated for classification.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples,)\n            Predicted target values for X, values are from ``classes_``\n        '
        check_is_fitted(self)
        K_star = self.kernel_(self.X_train_, X)
        f_star = K_star.T.dot(self.y_train_ - self.pi_)
        return np.where(f_star > 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        if False:
            return 10
        'Return probability estimates for the test vector X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Query points where the GP is evaluated for classification.\n\n        Returns\n        -------\n        C : array-like of shape (n_samples, n_classes)\n            Returns the probability of the samples for each class in\n            the model. The columns correspond to the classes in sorted\n            order, as they appear in the attribute ``classes_``.\n        '
        check_is_fitted(self)
        K_star = self.kernel_(self.X_train_, X)
        f_star = K_star.T.dot(self.y_train_ - self.pi_)
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)
        var_f_star = self.kernel_.diag(X) - np.einsum('ij,ij->j', v, v)
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS ** 2))) / (2 * np.sqrt(var_f_star * 2 * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()
        return np.vstack((1 - pi_star, pi_star)).T

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        if False:
            i = 10
            return i + 15
        'Returns log-marginal likelihood of theta for training data.\n\n        Parameters\n        ----------\n        theta : array-like of shape (n_kernel_params,), default=None\n            Kernel hyperparameters for which the log-marginal likelihood is\n            evaluated. If None, the precomputed log_marginal_likelihood\n            of ``self.kernel_.theta`` is returned.\n\n        eval_gradient : bool, default=False\n            If True, the gradient of the log-marginal likelihood with respect\n            to the kernel hyperparameters at position theta is returned\n            additionally. If True, theta must not be None.\n\n        clone_kernel : bool, default=True\n            If True, the kernel attribute is copied. If False, the kernel\n            attribute is modified, but may result in a performance improvement.\n\n        Returns\n        -------\n        log_likelihood : float\n            Log-marginal likelihood of theta for training data.\n\n        log_likelihood_gradient : ndarray of shape (n_kernel_params,),                 optional\n            Gradient of the log-marginal likelihood with respect to the kernel\n            hyperparameters at position theta.\n            Only returned when `eval_gradient` is True.\n        '
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
        (Z, (pi, W_sr, L, b, a)) = self._posterior_mode(K, return_temporaries=True)
        if not eval_gradient:
            return Z
        d_Z = np.empty(theta.shape[0])
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))
        C = solve(L, W_sr[:, np.newaxis] * K)
        s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) * (pi * (1 - pi) * (1 - 2 * pi))
        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]
            s_1 = 0.5 * a.T.dot(C).dot(a) - 0.5 * R.T.ravel().dot(C.ravel())
            b = C.dot(self.y_train_ - pi)
            s_3 = b - K.dot(R.dot(b))
            d_Z[j] = s_1 + s_2.T.dot(s_3)
        return (Z, d_Z)

    def _posterior_mode(self, K, return_temporaries=False):
        if False:
            for i in range(10):
                print('nop')
        "Mode-finding for binary Laplace GPC and fixed kernel.\n\n        This approximates the posterior of the latent function values for given\n        inputs and target observations with a Gaussian approximation and uses\n        Newton's iteration to find the mode of this approximation.\n        "
        if self.warm_start and hasattr(self, 'f_cached') and (self.f_cached.shape == self.y_train_.shape):
            f = self.f_cached
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float64)
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            pi = expit(f)
            W = pi * (1 - pi)
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:, np.newaxis] * K
            B = np.eye(W.shape[0]) + W_sr_K * W_sr
            L = cholesky(B, lower=True)
            b = W * f + (self.y_train_ - pi)
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            f = K.dot(a)
            lml = -0.5 * a.T.dot(f) - np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum() - np.log(np.diag(L)).sum()
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml
        self.f_cached = f
        if return_temporaries:
            return (log_marginal_likelihood, (pi, W_sr, L, b, a))
        else:
            return log_marginal_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if False:
            for i in range(10):
                print('nop')
        if self.optimizer == 'fmin_l_bfgs_b':
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B', jac=True, bounds=bounds)
            _check_optimize_result('lbfgs', opt_res)
            (theta_opt, func_min) = (opt_res.x, opt_res.fun)
        elif callable(self.optimizer):
            (theta_opt, func_min) = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError('Unknown optimizer %s.' % self.optimizer)
        return (theta_opt, func_min)

class GaussianProcessClassifier(ClassifierMixin, BaseEstimator):
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting. Also kernel
        cannot be a `CompoundKernel`.

    optimizer : 'fmin_l_bfgs_b', callable or None, default='fmin_l_bfgs_b'
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'L-BFGS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.

    max_iter_predict : int, default=100
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    warm_start : bool, default=False
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization. See :term:`the Glossary
        <warm_start>`.

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    multi_class : {'one_vs_rest', 'one_vs_one'}, default='one_vs_rest'
        Specifies how multi-class classification problems are handled.
        Supported are 'one_vs_rest' and 'one_vs_one'. In 'one_vs_rest',
        one binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest. In 'one_vs_one', one
        binary Gaussian process classifier is fitted for each pair of classes,
        which is trained to separate these two classes. The predictions of
        these binary predictors are combined into multi-class predictions.
        Note that 'one_vs_one' does not support predicting probability
        estimates.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the specified
        multiclass problems are computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    base_estimator_ : ``Estimator`` instance
        The estimator instance that defines the likelihood function
        using the observed data.

    kernel_ : kernel instance
        The kernel used for prediction. In case of binary classification,
        the structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters. In case of multi-class
        classification, a CompoundKernel is returned which consists of the
        different kernels used in the one-versus-rest classifiers.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        The number of classes in the training data

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianProcessRegressor : Gaussian process regression (GPR).

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.83548752, 0.03228706, 0.13222543],
           [0.79064206, 0.06525643, 0.14410151]])
    """
    _parameter_constraints: dict = {'kernel': [Kernel, None], 'optimizer': [StrOptions({'fmin_l_bfgs_b'}), callable, None], 'n_restarts_optimizer': [Interval(Integral, 0, None, closed='left')], 'max_iter_predict': [Interval(Integral, 1, None, closed='left')], 'warm_start': ['boolean'], 'copy_X_train': ['boolean'], 'random_state': ['random_state'], 'multi_class': [StrOptions({'one_vs_rest', 'one_vs_one'})], 'n_jobs': [Integral, None]}

    def __init__(self, kernel=None, *, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
        if False:
            return 10
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        if False:
            while True:
                i = 10
        'Fit Gaussian process classification model.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Feature vectors or other representations of training data.\n\n        y : array-like of shape (n_samples,)\n            Target values, must be binary.\n\n        Returns\n        -------\n        self : object\n            Returns an instance of self.\n        '
        if isinstance(self.kernel, CompoundKernel):
            raise ValueError('kernel cannot be a CompoundKernel')
        if self.kernel is None or self.kernel.requires_vector_input:
            (X, y) = self._validate_data(X, y, multi_output=False, ensure_2d=True, dtype='numeric')
        else:
            (X, y) = self._validate_data(X, y, multi_output=False, ensure_2d=False, dtype=None)
        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(kernel=self.kernel, optimizer=self.optimizer, n_restarts_optimizer=self.n_restarts_optimizer, max_iter_predict=self.max_iter_predict, warm_start=self.warm_start, copy_X_train=self.copy_X_train, random_state=self.random_state)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError('GaussianProcessClassifier requires 2 or more distinct classes; got %d class (only class %s is present)' % (self.n_classes_, self.classes_[0]))
        if self.n_classes_ > 2:
            if self.multi_class == 'one_vs_rest':
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_, n_jobs=self.n_jobs)
            elif self.multi_class == 'one_vs_one':
                self.base_estimator_ = OneVsOneClassifier(self.base_estimator_, n_jobs=self.n_jobs)
            else:
                raise ValueError('Unknown multi-class mode %s' % self.multi_class)
        self.base_estimator_.fit(X, y)
        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean([estimator.log_marginal_likelihood() for estimator in self.base_estimator_.estimators_])
        else:
            self.log_marginal_likelihood_value_ = self.base_estimator_.log_marginal_likelihood()
        return self

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Perform classification on an array of test vectors X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Query points where the GP is evaluated for classification.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples,)\n            Predicted target values for X, values are from ``classes_``.\n        '
        check_is_fitted(self)
        if self.kernel is None or self.kernel.requires_vector_input:
            X = self._validate_data(X, ensure_2d=True, dtype='numeric', reset=False)
        else:
            X = self._validate_data(X, ensure_2d=False, dtype=None, reset=False)
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        if False:
            return 10
        'Return probability estimates for the test vector X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or list of object\n            Query points where the GP is evaluated for classification.\n\n        Returns\n        -------\n        C : array-like of shape (n_samples, n_classes)\n            Returns the probability of the samples for each class in\n            the model. The columns correspond to the classes in sorted\n            order, as they appear in the attribute :term:`classes_`.\n        '
        check_is_fitted(self)
        if self.n_classes_ > 2 and self.multi_class == 'one_vs_one':
            raise ValueError('one_vs_one multi-class mode does not support predicting probability estimates. Use one_vs_rest mode instead.')
        if self.kernel is None or self.kernel.requires_vector_input:
            X = self._validate_data(X, ensure_2d=True, dtype='numeric', reset=False)
        else:
            X = self._validate_data(X, ensure_2d=False, dtype=None, reset=False)
        return self.base_estimator_.predict_proba(X)

    @property
    def kernel_(self):
        if False:
            i = 10
            return i + 15
        'Return the kernel of the base estimator.'
        if self.n_classes_ == 2:
            return self.base_estimator_.kernel_
        else:
            return CompoundKernel([estimator.kernel_ for estimator in self.base_estimator_.estimators_])

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        if False:
            print('Hello World!')
        'Return log-marginal likelihood of theta for training data.\n\n        In the case of multi-class classification, the mean log-marginal\n        likelihood of the one-versus-rest classifiers are returned.\n\n        Parameters\n        ----------\n        theta : array-like of shape (n_kernel_params,), default=None\n            Kernel hyperparameters for which the log-marginal likelihood is\n            evaluated. In the case of multi-class classification, theta may\n            be the  hyperparameters of the compound kernel or of an individual\n            kernel. In the latter case, all individual kernel get assigned the\n            same theta values. If None, the precomputed log_marginal_likelihood\n            of ``self.kernel_.theta`` is returned.\n\n        eval_gradient : bool, default=False\n            If True, the gradient of the log-marginal likelihood with respect\n            to the kernel hyperparameters at position theta is returned\n            additionally. Note that gradient computation is not supported\n            for non-binary classification. If True, theta must not be None.\n\n        clone_kernel : bool, default=True\n            If True, the kernel attribute is copied. If False, the kernel\n            attribute is modified, but may result in a performance improvement.\n\n        Returns\n        -------\n        log_likelihood : float\n            Log-marginal likelihood of theta for training data.\n\n        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional\n            Gradient of the log-marginal likelihood with respect to the kernel\n            hyperparameters at position theta.\n            Only returned when `eval_gradient` is True.\n        '
        check_is_fitted(self)
        if theta is None:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated for theta!=None')
            return self.log_marginal_likelihood_value_
        theta = np.asarray(theta)
        if self.n_classes_ == 2:
            return self.base_estimator_.log_marginal_likelihood(theta, eval_gradient, clone_kernel=clone_kernel)
        else:
            if eval_gradient:
                raise NotImplementedError('Gradient of log-marginal-likelihood not implemented for multi-class GPC.')
            estimators = self.base_estimator_.estimators_
            n_dims = estimators[0].kernel_.n_dims
            if theta.shape[0] == n_dims:
                return np.mean([estimator.log_marginal_likelihood(theta, clone_kernel=clone_kernel) for (i, estimator) in enumerate(estimators)])
            elif theta.shape[0] == n_dims * self.classes_.shape[0]:
                return np.mean([estimator.log_marginal_likelihood(theta[n_dims * i:n_dims * (i + 1)], clone_kernel=clone_kernel) for (i, estimator) in enumerate(estimators)])
            else:
                raise ValueError('Shape of theta must be either %d or %d. Obtained theta with shape %d.' % (n_dims, n_dims * self.classes_.shape[0], theta.shape[0]))