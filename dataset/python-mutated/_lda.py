"""

=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: https://github.com/blei-lab/onlineldavb
"""
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from ..utils import check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Interval, StrOptions
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._online_lda_fast import _dirichlet_expectation_1d as cy_dirichlet_expectation_1d
from ._online_lda_fast import _dirichlet_expectation_2d
from ._online_lda_fast import mean_change as cy_mean_change
EPS = np.finfo(float).eps

def _update_doc_distribution(X, exp_topic_word_distr, doc_topic_prior, max_doc_update_iter, mean_change_tol, cal_sstats, random_state):
    if False:
        for i in range(10):
            print('nop')
    'E-step: update document-topic distribution.\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        Document word matrix.\n\n    exp_topic_word_distr : ndarray of shape (n_topics, n_features)\n        Exponential value of expectation of log topic word distribution.\n        In the literature, this is `exp(E[log(beta)])`.\n\n    doc_topic_prior : float\n        Prior of document topic distribution `theta`.\n\n    max_doc_update_iter : int\n        Max number of iterations for updating document topic distribution in\n        the E-step.\n\n    mean_change_tol : float\n        Stopping tolerance for updating document topic distribution in E-step.\n\n    cal_sstats : bool\n        Parameter that indicate to calculate sufficient statistics or not.\n        Set `cal_sstats` to `True` when we need to run M-step.\n\n    random_state : RandomState instance or None\n        Parameter that indicate how to initialize document topic distribution.\n        Set `random_state` to None will initialize document topic distribution\n        to a constant number.\n\n    Returns\n    -------\n    (doc_topic_distr, suff_stats) :\n        `doc_topic_distr` is unnormalized topic distribution for each document.\n        In the literature, this is `gamma`. we can calculate `E[log(theta)]`\n        from it.\n        `suff_stats` is expected sufficient statistics for the M-step.\n            When `cal_sstats == False`, this will be None.\n\n    '
    is_sparse_x = sp.issparse(X)
    (n_samples, n_features) = X.shape
    n_topics = exp_topic_word_distr.shape[0]
    if random_state:
        doc_topic_distr = random_state.gamma(100.0, 0.01, (n_samples, n_topics)).astype(X.dtype, copy=False)
    else:
        doc_topic_distr = np.ones((n_samples, n_topics), dtype=X.dtype)
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))
    suff_stats = np.zeros(exp_topic_word_distr.shape, dtype=X.dtype) if cal_sstats else None
    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr
    ctype = 'float' if X.dtype == np.float32 else 'double'
    mean_change = cy_mean_change[ctype]
    dirichlet_expectation_1d = cy_dirichlet_expectation_1d[ctype]
    eps = np.finfo(X.dtype).eps
    for idx_d in range(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]
        doc_topic_d = doc_topic_distr[idx_d, :]
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        exp_topic_word_d = exp_topic_word_distr[:, ids]
        for _ in range(0, max_doc_update_iter):
            last_d = doc_topic_d
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps
            doc_topic_d = exp_doc_topic_d * np.dot(cnts / norm_phi, exp_topic_word_d.T)
            dirichlet_expectation_1d(doc_topic_d, doc_topic_prior, exp_doc_topic_d)
            if mean_change(last_d, doc_topic_d) < mean_change_tol:
                break
        doc_topic_distr[idx_d, :] = doc_topic_d
        if cal_sstats:
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps
            suff_stats[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)
    return (doc_topic_distr, suff_stats)

class LatentDirichletAllocation(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Latent Dirichlet Allocation with online variational Bayes algorithm.

    The implementation is based on [1]_ and [2]_.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <LatentDirichletAllocation>`.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.

        .. versionchanged:: 0.19
            ``n_topics`` was renamed to ``n_components``

    doc_topic_prior : float, default=None
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_components`.
        In [1]_, this is called `alpha`.

    topic_word_prior : float, default=None
        Prior of topic word distribution `beta`. If the value is None, defaults
        to `1 / n_components`.
        In [1]_, this is called `eta`.

    learning_method : {'batch', 'online'}, default='batch'
        Method used to update `_component`. Only used in :meth:`fit` method.
        In general, if the data size is large, the online update will be much
        faster than the batch update.

        Valid options::

            'batch': Batch variational Bayes method. Use all training data in
                each EM update.
                Old `components_` will be overwritten in each iteration.
            'online': Online variational Bayes method. In each EM update, use
                mini-batch of training data to update the ``components_``
                variable incrementally. The learning rate is controlled by the
                ``learning_decay`` and the ``learning_offset`` parameters.

        .. versionchanged:: 0.20
            The default learning method is now ``"batch"``.

    learning_decay : float, default=0.7
        It is a parameter that control learning rate in the online learning
        method. The value should be set between (0.5, 1.0] to guarantee
        asymptotic convergence. When the value is 0.0 and batch_size is
        ``n_samples``, the update method is same as batch learning. In the
        literature, this is called kappa.

    learning_offset : float, default=10.0
        A (positive) parameter that downweights early iterations in online
        learning.  It should be greater than 1.0. In the literature, this is
        called tau_0.

    max_iter : int, default=10
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the :meth:`fit` method, and not the
        :meth:`partial_fit` method.

    batch_size : int, default=128
        Number of documents to use in each EM iteration. Only used in online
        learning.

    evaluate_every : int, default=-1
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.

    total_samples : int, default=1e6
        Total number of documents. Only used in the :meth:`partial_fit` method.

    perp_tol : float, default=1e-1
        Perplexity tolerance in batch learning. Only used when
        ``evaluate_every`` is greater than 0.

    mean_change_tol : float, default=1e-3
        Stopping tolerance for updating document topic distribution in E-step.

    max_doc_update_iter : int, default=100
        Max number of iterations for updating document topic distribution in
        the E-step.

    n_jobs : int, default=None
        The number of jobs to use in the E-step.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Variational parameters for topic word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j]`` can be viewed as pseudocount that represents the
        number of times word `j` was assigned to topic `i`.
        It can also be viewed as distribution over the words for each topic
        after normalization:
        ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

    exp_dirichlet_component_ : ndarray of shape (n_components, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    n_batch_iter_ : int
        Number of iterations of the EM step.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of passes over the dataset.

    bound_ : float
        Final perplexity score on training set.

    doc_topic_prior_ : float
        Prior of document topic distribution `theta`. If the value is None,
        it is `1 / n_components`.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generator or by `np.random`.

    topic_word_prior_ : float
        Prior of topic word distribution `beta`. If the value is None, it is
        `1 / n_components`.

    See Also
    --------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
        A classifier with a linear decision boundary, generated by fitting
        class conditional densities to the data and using Bayes' rule.

    References
    ----------
    .. [1] "Online Learning for Latent Dirichlet Allocation", Matthew D.
           Hoffman, David M. Blei, Francis Bach, 2010
           https://github.com/blei-lab/onlineldavb

    .. [2] "Stochastic Variational Inference", Matthew D. Hoffman,
           David M. Blei, Chong Wang, John Paisley, 2013

    Examples
    --------
    >>> from sklearn.decomposition import LatentDirichletAllocation
    >>> from sklearn.datasets import make_multilabel_classification
    >>> # This produces a feature matrix of token counts, similar to what
    >>> # CountVectorizer would produce on text.
    >>> X, _ = make_multilabel_classification(random_state=0)
    >>> lda = LatentDirichletAllocation(n_components=5,
    ...     random_state=0)
    >>> lda.fit(X)
    LatentDirichletAllocation(...)
    >>> # get topics for some given samples:
    >>> lda.transform(X[-2:])
    array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
           [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 0, None, closed='neither')], 'doc_topic_prior': [None, Interval(Real, 0, 1, closed='both')], 'topic_word_prior': [None, Interval(Real, 0, 1, closed='both')], 'learning_method': [StrOptions({'batch', 'online'})], 'learning_decay': [Interval(Real, 0, 1, closed='both')], 'learning_offset': [Interval(Real, 1.0, None, closed='left')], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'batch_size': [Interval(Integral, 0, None, closed='neither')], 'evaluate_every': [Interval(Integral, None, None, closed='neither')], 'total_samples': [Interval(Real, 0, None, closed='neither')], 'perp_tol': [Interval(Real, 0, None, closed='left')], 'mean_change_tol': [Interval(Real, 0, None, closed='left')], 'max_doc_update_iter': [Interval(Integral, 0, None, closed='left')], 'n_jobs': [None, Integral], 'verbose': ['verbose'], 'random_state': ['random_state']}

    def __init__(self, n_components=10, *, doc_topic_prior=None, topic_word_prior=None, learning_method='batch', learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None):
        if False:
            i = 10
            return i + 15
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _init_latent_vars(self, n_features, dtype=np.float64):
        if False:
            i = 10
            return i + 15
        'Initialize latent variables.'
        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0
        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior
        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior
        init_gamma = 100.0
        init_var = 1.0 / init_gamma
        self.components_ = self.random_state_.gamma(init_gamma, init_var, (self.n_components, n_features)).astype(dtype, copy=False)
        self.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(self.components_))

    def _e_step(self, X, cal_sstats, random_init, parallel=None):
        if False:
            i = 10
            return i + 15
        'E-step in EM update.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        cal_sstats : bool\n            Parameter that indicate whether to calculate sufficient statistics\n            or not. Set ``cal_sstats`` to True when we need to run M-step.\n\n        random_init : bool\n            Parameter that indicate whether to initialize document topic\n            distribution randomly in the E-step. Set it to True in training\n            steps.\n\n        parallel : joblib.Parallel, default=None\n            Pre-initialized instance of joblib.Parallel.\n\n        Returns\n        -------\n        (doc_topic_distr, suff_stats) :\n            `doc_topic_distr` is unnormalized topic distribution for each\n            document. In the literature, this is called `gamma`.\n            `suff_stats` is expected sufficient statistics for the M-step.\n            When `cal_sstats == False`, it will be None.\n\n        '
        random_state = self.random_state_ if random_init else None
        n_jobs = effective_n_jobs(self.n_jobs)
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))
        results = parallel((delayed(_update_doc_distribution)(X[idx_slice, :], self.exp_dirichlet_component_, self.doc_topic_prior_, self.max_doc_update_iter, self.mean_change_tol, cal_sstats, random_state) for idx_slice in gen_even_slices(X.shape[0], n_jobs)))
        (doc_topics, sstats_list) = zip(*results)
        doc_topic_distr = np.vstack(doc_topics)
        if cal_sstats:
            suff_stats = np.zeros(self.components_.shape, dtype=self.components_.dtype)
            for sstats in sstats_list:
                suff_stats += sstats
            suff_stats *= self.exp_dirichlet_component_
        else:
            suff_stats = None
        return (doc_topic_distr, suff_stats)

    def _em_step(self, X, total_samples, batch_update, parallel=None):
        if False:
            while True:
                i = 10
        'EM update for 1 iteration.\n\n        update `_component` by batch VB or online VB.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        total_samples : int\n            Total number of documents. It is only used when\n            batch_update is `False`.\n\n        batch_update : bool\n            Parameter that controls updating method.\n            `True` for batch learning, `False` for online learning.\n\n        parallel : joblib.Parallel, default=None\n            Pre-initialized instance of joblib.Parallel\n\n        Returns\n        -------\n        doc_topic_distr : ndarray of shape (n_samples, n_components)\n            Unnormalized document topic distribution.\n        '
        (_, suff_stats) = self._e_step(X, cal_sstats=True, random_init=True, parallel=parallel)
        if batch_update:
            self.components_ = self.topic_word_prior_ + suff_stats
        else:
            weight = np.power(self.learning_offset + self.n_batch_iter_, -self.learning_decay)
            doc_ratio = float(total_samples) / X.shape[0]
            self.components_ *= 1 - weight
            self.components_ += weight * (self.topic_word_prior_ + doc_ratio * suff_stats)
        self.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(self.components_))
        self.n_batch_iter_ += 1
        return

    def _more_tags(self):
        if False:
            return 10
        return {'preserves_dtype': [np.float64, np.float32], 'requires_positive_X': True}

    def _check_non_neg_array(self, X, reset_n_features, whom):
        if False:
            print('Hello World!')
        'check X format\n\n        check X format and make sure no negative value in X.\n\n        Parameters\n        ----------\n        X :  array-like or sparse matrix\n\n        '
        dtype = [np.float64, np.float32] if reset_n_features else self.components_.dtype
        X = self._validate_data(X, reset=reset_n_features, accept_sparse='csr', dtype=dtype)
        check_non_negative(X, whom)
        return X

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Online VB with Mini-Batch update.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        self\n            Partially fitted estimator.\n        '
        first_time = not hasattr(self, 'components_')
        X = self._check_non_neg_array(X, reset_n_features=first_time, whom='LatentDirichletAllocation.partial_fit')
        (n_samples, n_features) = X.shape
        batch_size = self.batch_size
        if first_time:
            self._init_latent_vars(n_features, dtype=X.dtype)
        if n_features != self.components_.shape[1]:
            raise ValueError('The provided data has %d dimensions while the model was trained with feature size %d.' % (n_features, self.components_.shape[1]))
        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for idx_slice in gen_batches(n_samples, batch_size):
                self._em_step(X[idx_slice, :], total_samples=self.total_samples, batch_update=False, parallel=parallel)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        "Learn model for the data X with variational Bayes method.\n\n        When `learning_method` is 'online', use mini-batch update.\n        Otherwise, use batch update.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        self\n            Fitted estimator.\n        "
        X = self._check_non_neg_array(X, reset_n_features=True, whom='LatentDirichletAllocation.fit')
        (n_samples, n_features) = X.shape
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every
        learning_method = self.learning_method
        batch_size = self.batch_size
        self._init_latent_vars(n_features, dtype=X.dtype)
        last_bound = None
        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for i in range(max_iter):
                if learning_method == 'online':
                    for idx_slice in gen_batches(n_samples, batch_size):
                        self._em_step(X[idx_slice, :], total_samples=n_samples, batch_update=False, parallel=parallel)
                else:
                    self._em_step(X, total_samples=n_samples, batch_update=True, parallel=parallel)
                if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                    (doc_topics_distr, _) = self._e_step(X, cal_sstats=False, random_init=False, parallel=parallel)
                    bound = self._perplexity_precomp_distr(X, doc_topics_distr, sub_sampling=False)
                    if self.verbose:
                        print('iteration: %d of max_iter: %d, perplexity: %.4f' % (i + 1, max_iter, bound))
                    if last_bound and abs(last_bound - bound) < self.perp_tol:
                        break
                    last_bound = bound
                elif self.verbose:
                    print('iteration: %d of max_iter: %d' % (i + 1, max_iter))
                self.n_iter_ += 1
        (doc_topics_distr, _) = self._e_step(X, cal_sstats=False, random_init=False, parallel=parallel)
        self.bound_ = self._perplexity_precomp_distr(X, doc_topics_distr, sub_sampling=False)
        return self

    def _unnormalized_transform(self, X):
        if False:
            print('Hello World!')
        'Transform data X according to fitted model.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        Returns\n        -------\n        doc_topic_distr : ndarray of shape (n_samples, n_components)\n            Document topic distribution for X.\n        '
        (doc_topic_distr, _) = self._e_step(X, cal_sstats=False, random_init=False)
        return doc_topic_distr

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Transform data X according to the fitted model.\n\n           .. versionchanged:: 0.18\n              *doc_topic_distr* is now normalized\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        Returns\n        -------\n        doc_topic_distr : ndarray of shape (n_samples, n_components)\n            Document topic distribution for X.\n        '
        check_is_fitted(self)
        X = self._check_non_neg_array(X, reset_n_features=False, whom='LatentDirichletAllocation.transform')
        doc_topic_distr = self._unnormalized_transform(X)
        doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        return doc_topic_distr

    def _approx_bound(self, X, doc_topic_distr, sub_sampling):
        if False:
            while True:
                i = 10
        'Estimate the variational bound.\n\n        Estimate the variational bound over "all documents" using only the\n        documents passed in as X. Since log-likelihood of each word cannot\n        be computed directly, we use this bound to estimate it.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        doc_topic_distr : ndarray of shape (n_samples, n_components)\n            Document topic distribution. In the literature, this is called\n            gamma.\n\n        sub_sampling : bool, default=False\n            Compensate for subsampling of documents.\n            It is used in calculate bound in online learning.\n\n        Returns\n        -------\n        score : float\n\n        '

        def _loglikelihood(prior, distr, dirichlet_distr, size):
            if False:
                print('Hello World!')
            score = np.sum((prior - distr) * dirichlet_distr)
            score += np.sum(gammaln(distr) - gammaln(prior))
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
            return score
        is_sparse_x = sp.issparse(X)
        (n_samples, n_components) = doc_topic_distr.shape
        n_features = self.components_.shape[1]
        score = 0
        dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic_distr)
        dirichlet_component_ = _dirichlet_expectation_2d(self.components_)
        doc_topic_prior = self.doc_topic_prior_
        topic_word_prior = self.topic_word_prior_
        if is_sparse_x:
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr
        for idx_d in range(0, n_samples):
            if is_sparse_x:
                ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
                cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            else:
                ids = np.nonzero(X[idx_d, :])[0]
                cnts = X[idx_d, ids]
            temp = dirichlet_doc_topic[idx_d, :, np.newaxis] + dirichlet_component_[:, ids]
            norm_phi = logsumexp(temp, axis=0)
            score += np.dot(cnts, norm_phi)
        score += _loglikelihood(doc_topic_prior, doc_topic_distr, dirichlet_doc_topic, self.n_components)
        if sub_sampling:
            doc_ratio = float(self.total_samples) / n_samples
            score *= doc_ratio
        score += _loglikelihood(topic_word_prior, self.components_, dirichlet_component_, n_features)
        return score

    def score(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Calculate approximate log-likelihood as score.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        score : float\n            Use approximate bound as score.\n        '
        check_is_fitted(self)
        X = self._check_non_neg_array(X, reset_n_features=False, whom='LatentDirichletAllocation.score')
        doc_topic_distr = self._unnormalized_transform(X)
        score = self._approx_bound(X, doc_topic_distr, sub_sampling=False)
        return score

    def _perplexity_precomp_distr(self, X, doc_topic_distr=None, sub_sampling=False):
        if False:
            while True:
                i = 10
        'Calculate approximate perplexity for data X with ability to accept\n        precomputed doc_topic_distr\n\n        Perplexity is defined as exp(-1. * log-likelihood per word)\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        doc_topic_distr : ndarray of shape (n_samples, n_components),                 default=None\n            Document topic distribution.\n            If it is None, it will be generated by applying transform on X.\n\n        Returns\n        -------\n        score : float\n            Perplexity score.\n        '
        if doc_topic_distr is None:
            doc_topic_distr = self._unnormalized_transform(X)
        else:
            (n_samples, n_components) = doc_topic_distr.shape
            if n_samples != X.shape[0]:
                raise ValueError('Number of samples in X and doc_topic_distr do not match.')
            if n_components != self.n_components:
                raise ValueError('Number of topics does not match.')
        current_samples = X.shape[0]
        bound = self._approx_bound(X, doc_topic_distr, sub_sampling)
        if sub_sampling:
            word_cnt = X.sum() * (float(self.total_samples) / current_samples)
        else:
            word_cnt = X.sum()
        perword_bound = bound / word_cnt
        return np.exp(-1.0 * perword_bound)

    def perplexity(self, X, sub_sampling=False):
        if False:
            i = 10
            return i + 15
        'Calculate approximate perplexity for data X.\n\n        Perplexity is defined as exp(-1. * log-likelihood per word)\n\n        .. versionchanged:: 0.19\n           *doc_topic_distr* argument has been deprecated and is ignored\n           because user no longer has access to unnormalized distribution\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Document word matrix.\n\n        sub_sampling : bool\n            Do sub-sampling or not.\n\n        Returns\n        -------\n        score : float\n            Perplexity score.\n        '
        check_is_fitted(self)
        X = self._check_non_neg_array(X, reset_n_features=True, whom='LatentDirichletAllocation.perplexity')
        return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)

    @property
    def _n_features_out(self):
        if False:
            return 10
        'Number of transformed output features.'
        return self.components_.shape[0]