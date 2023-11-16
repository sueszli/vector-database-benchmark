"""Restricted Boltzmann Machine
"""
import time
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from ..utils import check_random_state, gen_even_slices
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted

class BernoulliRBM(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, default=256
        Number of binary hidden units.

    learning_rate : float, default=0.1
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, default=10
        Number of examples per minibatch.

    n_iter : int, default=10
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, default=0
        The verbosity level. The default, zero, means silent mode. Range
        of values is [0, inf].

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for:

        - Gibbs sampling from visible and hidden layers.

        - Initializing components, sampling from layers during fit.

        - Corrupting the data when scoring samples.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    intercept_hidden_ : array-like of shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like of shape (n_features,)
        Biases of the visible units.

    components_ : array-like of shape (n_components, n_features)
        Weight matrix, where `n_features` is the number of
        visible units and `n_components` is the number of hidden units.

    h_samples_ : array-like of shape (batch_size, n_components)
        Hidden Activation sampled from the model distribution,
        where `batch_size` is the number of examples per minibatch and
        `n_components` is the number of hidden units.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
    sklearn.decomposition.PCA : An unsupervised linear dimensionality
        reduction model.

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(n_components=2)
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'learning_rate': [Interval(Real, 0, None, closed='neither')], 'batch_size': [Interval(Integral, 1, None, closed='left')], 'n_iter': [Interval(Integral, 0, None, closed='left')], 'verbose': ['verbose'], 'random_state': ['random_state']}

    def __init__(self, n_components=256, *, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None):
        if False:
            while True:
                i = 10
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def transform(self, X):
        if False:
            return 10
        'Compute the hidden layer activation probabilities, P(h=1|v=X).\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The data to be transformed.\n\n        Returns\n        -------\n        h : ndarray of shape (n_samples, n_components)\n            Latent representations of the data.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse='csr', reset=False, dtype=(np.float64, np.float32))
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        if False:
            print('Hello World!')
        'Computes the probabilities P(h=1|v).\n\n        Parameters\n        ----------\n        v : ndarray of shape (n_samples, n_features)\n            Values of the visible layer.\n\n        Returns\n        -------\n        h : ndarray of shape (n_samples, n_components)\n            Corresponding mean field values for the hidden layer.\n        '
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)

    def _sample_hiddens(self, v, rng):
        if False:
            return 10
        'Sample from the distribution P(h|v).\n\n        Parameters\n        ----------\n        v : ndarray of shape (n_samples, n_features)\n            Values of the visible layer to sample from.\n\n        rng : RandomState instance\n            Random number generator to use.\n\n        Returns\n        -------\n        h : ndarray of shape (n_samples, n_components)\n            Values of the hidden layer.\n        '
        p = self._mean_hiddens(v)
        return rng.uniform(size=p.shape) < p

    def _sample_visibles(self, h, rng):
        if False:
            print('Hello World!')
        'Sample from the distribution P(v|h).\n\n        Parameters\n        ----------\n        h : ndarray of shape (n_samples, n_components)\n            Values of the hidden layer to sample from.\n\n        rng : RandomState instance\n            Random number generator to use.\n\n        Returns\n        -------\n        v : ndarray of shape (n_samples, n_features)\n            Values of the visible layer.\n        '
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return rng.uniform(size=p.shape) < p

    def _free_energy(self, v):
        if False:
            for i in range(10):
                print('nop')
        'Computes the free energy F(v) = - log sum_h exp(-E(v,h)).\n\n        Parameters\n        ----------\n        v : ndarray of shape (n_samples, n_features)\n            Values of the visible layer.\n\n        Returns\n        -------\n        free_energy : ndarray of shape (n_samples,)\n            The value of the free energy.\n        '
        return -safe_sparse_dot(v, self.intercept_visible_) - np.logaddexp(0, safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_).sum(axis=1)

    def gibbs(self, v):
        if False:
            while True:
                i = 10
        'Perform one Gibbs sampling step.\n\n        Parameters\n        ----------\n        v : ndarray of shape (n_samples, n_features)\n            Values of the visible layer to start from.\n\n        Returns\n        -------\n        v_new : ndarray of shape (n_samples, n_features)\n            Values of the visible layer after one Gibbs step.\n        '
        check_is_fitted(self)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)
        return v_

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        if False:
            return 10
        'Fit the model to the partial segment of the data X.\n\n        Parameters\n        ----------\n        X : ndarray of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None\n            Target values (None for unsupervised transformations).\n\n        Returns\n        -------\n        self : BernoulliRBM\n            The fitted model.\n        '
        first_pass = not hasattr(self, 'components_')
        X = self._validate_data(X, accept_sparse='csr', dtype=np.float64, reset=first_pass)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(self.random_state_.normal(0, 0.01, (self.n_components, X.shape[1])), order='F')
            self._n_features_out = self.components_.shape[0]
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components)
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1])
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))
        self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        if False:
            for i in range(10):
                print('nop')
        'Inner fit for one mini-batch.\n\n        Adjust the parameters to maximize the likelihood of v using\n        Stochastic Maximum Likelihood (SML).\n\n        Parameters\n        ----------\n        v_pos : ndarray of shape (n_samples, n_features)\n            The data to use for training.\n\n        rng : RandomState instance\n            Random number generator to use for sampling.\n        '
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)
        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0))
        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score_samples(self, X):
        if False:
            print('Hello World!')
        'Compute the pseudo-likelihood of X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Values of the visible layer. Must be all-boolean (not checked).\n\n        Returns\n        -------\n        pseudo_likelihood : ndarray of shape (n_samples,)\n            Value of the pseudo-likelihood (proxy for likelihood).\n\n        Notes\n        -----\n        This method is not deterministic: it computes a quantity called the\n        free energy on X, then on a randomly corrupted version of X, and\n        returns the log of the logistic function of the difference.\n        '
        check_is_fitted(self)
        v = self._validate_data(X, accept_sparse='csr', reset=False)
        rng = check_random_state(self.random_state)
        ind = (np.arange(v.shape[0]), rng.randint(0, v.shape[1], v.shape[0]))
        if sp.issparse(v):
            data = -2 * v[ind] + 1
            if isinstance(data, np.matrix):
                v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
            else:
                v_ = v + sp.csr_array((data.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]
        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return -v.shape[1] * np.logaddexp(0, -(fe_ - fe))

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Fit the model to the data X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None\n            Target values (None for unsupervised transformations).\n\n        Returns\n        -------\n        self : BernoulliRBM\n            The fitted model.\n        '
        X = self._validate_data(X, accept_sparse='csr', dtype=(np.float64, np.float32))
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)
        self.components_ = np.asarray(rng.normal(0, 0.01, (self.n_components, X.shape[1])), order='F', dtype=X.dtype)
        self._n_features_out = self.components_.shape[0]
        self.intercept_hidden_ = np.zeros(self.n_components, dtype=X.dtype)
        self.intercept_visible_ = np.zeros(X.shape[1], dtype=X.dtype)
        self.h_samples_ = np.zeros((self.batch_size, self.n_components), dtype=X.dtype)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)
            if verbose:
                end = time.time()
                print('[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs' % (type(self).__name__, iteration, self.score_samples(X).mean(), end - begin))
                begin = end
        return self

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'_xfail_checks': {'check_methods_subset_invariance': 'fails for the decision_function method', 'check_methods_sample_order_invariance': 'fails for the score_samples method'}, 'preserves_dtype': [np.float64, np.float32]}