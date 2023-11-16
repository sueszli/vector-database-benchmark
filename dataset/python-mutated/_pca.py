""" Principal Component Analysis.
"""
from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.special import gammaln
from ..base import _fit_context
from ..utils import check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._array_api import _convert_to_numpy, get_namespace
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.deprecation import deprecated
from ..utils.extmath import fast_logdet, randomized_svd, stable_cumsum, svd_flip
from ..utils.sparsefuncs import _implicit_column_offset, mean_variance_axis
from ..utils.validation import check_is_fitted
from ._base import _BasePCA

def _assess_dimension(spectrum, rank, n_samples):
    if False:
        while True:
            i = 10
    "Compute the log-likelihood of a rank ``rank`` dataset.\n\n    The dataset is assumed to be embedded in gaussian noise of shape(n,\n    dimf) having spectrum ``spectrum``. This implements the method of\n    T. P. Minka.\n\n    Parameters\n    ----------\n    spectrum : ndarray of shape (n_features,)\n        Data spectrum.\n    rank : int\n        Tested rank value. It should be strictly lower than n_features,\n        otherwise the method isn't specified (division by zero in equation\n        (31) from the paper).\n    n_samples : int\n        Number of samples.\n\n    Returns\n    -------\n    ll : float\n        The log-likelihood.\n\n    References\n    ----------\n    This implements the method of `Thomas P. Minka:\n    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604\n    <https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf>`_\n    "
    (xp, _) = get_namespace(spectrum)
    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:
        raise ValueError('the tested rank should be in [1, n_features - 1]')
    eps = 1e-15
    if spectrum[rank - 1] < eps:
        return -xp.inf
    pu = -rank * log(2.0)
    for i in range(1, rank + 1):
        pu += gammaln((n_features - i + 1) / 2.0) - log(xp.pi) * (n_features - i + 1) / 2.0
    pl = xp.sum(xp.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.0
    v = max(eps, xp.sum(spectrum[rank:]) / (n_features - rank))
    pv = -log(v) * n_samples * (n_features - rank) / 2.0
    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = log(2.0 * xp.pi) * (m + rank) / 2.0
    pa = 0.0
    spectrum_ = xp.asarray(spectrum, copy=True)
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, spectrum.shape[0]):
            pa += log((spectrum[i] - spectrum[j]) * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])) + log(n_samples)
    ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0
    return ll

def _infer_dimension(spectrum, n_samples):
    if False:
        i = 10
        return i + 15
    'Infers the dimension of a dataset with a given spectrum.\n\n    The returned value will be in [1, n_features - 1].\n    '
    (xp, _) = get_namespace(spectrum)
    ll = xp.empty_like(spectrum)
    ll[0] = -xp.inf
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return xp.argmax(ll)

class PCA(_BasePCA):
    """Principal component analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    It uses the LAPACK implementation of the full SVD or a randomized truncated
    SVD by the method of Halko et al. 2009, depending on the shape of the input
    data and the number of components to extract.

    It can also use the scipy.sparse.linalg ARPACK implementation of the
    truncated SVD.

    Notice that this class does not support sparse input. See
    :class:`TruncatedSVD` for an alternative with sparse data.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

        .. versionadded:: 0.18.0

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

        .. versionadded:: 0.18.0

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

        .. versionadded:: 0.18.0

    n_oversamples : int, default=10
        This parameter is only relevant when `svd_solver="randomized"`.
        It corresponds to the additional number of random vectors to sample the
        range of `X` so as to ensure proper conditioning. See
        :func:`~sklearn.utils.extmath.randomized_svd` for more details.

        .. versionadded:: 1.1

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
        for more details.

        .. versionadded:: 1.1

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 0.18.0

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. Equivalently, the right singular
        vectors of the centered input data, parallel to its eigenvectors.
        The components are sorted by decreasing ``explained_variance_``.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        The variance estimation uses `n_samples - 1` degrees of freedom.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

        .. versionadded:: 0.18

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

        .. versionadded:: 0.19

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KernelPCA : Kernel Principal Component Analysis.
    SparsePCA : Sparse Principal Component Analysis.
    TruncatedSVD : Dimensionality reduction using truncated SVD.
    IncrementalPCA : Incremental Principal Component Analysis.

    References
    ----------
    For n_components == 'mle', this class uses the method from:
    `Minka, T. P.. "Automatic choice of dimensionality for PCA".
    In NIPS, pp. 598-604 <https://tminka.github.io/papers/pca/minka-pca.pdf>`_

    Implements the probabilistic PCA model from:
    `Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
    component analysis". Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 61(3), 611-622.
    <http://www.miketipping.com/papers/met-mppca.pdf>`_
    via the score and score_samples methods.

    For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

    For svd_solver == 'randomized', see:
    :doi:`Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions".
    SIAM review, 53(2), 217-288.
    <10.1137/090771806>`
    and also
    :doi:`Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
    "A randomized algorithm for the decomposition of matrices".
    Applied and Computational Harmonic Analysis, 30(1), 47-68.
    <10.1016/j.acha.2010.02.003>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(n_components=2)
    >>> print(pca.explained_variance_ratio_)
    [0.9924... 0.0075...]
    >>> print(pca.singular_values_)
    [6.30061... 0.54980...]

    >>> pca = PCA(n_components=2, svd_solver='full')
    >>> pca.fit(X)
    PCA(n_components=2, svd_solver='full')
    >>> print(pca.explained_variance_ratio_)
    [0.9924... 0.00755...]
    >>> print(pca.singular_values_)
    [6.30061... 0.54980...]

    >>> pca = PCA(n_components=1, svd_solver='arpack')
    >>> pca.fit(X)
    PCA(n_components=1, svd_solver='arpack')
    >>> print(pca.explained_variance_ratio_)
    [0.99244...]
    >>> print(pca.singular_values_)
    [6.30061...]
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 0, None, closed='left'), Interval(RealNotInt, 0, 1, closed='neither'), StrOptions({'mle'}), None], 'copy': ['boolean'], 'whiten': ['boolean'], 'svd_solver': [StrOptions({'auto', 'full', 'arpack', 'randomized'})], 'tol': [Interval(Real, 0, None, closed='left')], 'iterated_power': [StrOptions({'auto'}), Interval(Integral, 0, None, closed='left')], 'n_oversamples': [Interval(Integral, 1, None, closed='left')], 'power_iteration_normalizer': [StrOptions({'auto', 'QR', 'LU', 'none'})], 'random_state': ['random_state']}

    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None):
        if False:
            print('Hello World!')
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    @deprecated('Attribute `n_features_` was deprecated in version 1.2 and will be removed in 1.4. Use `n_features_in_` instead.')
    @property
    def n_features_(self):
        if False:
            return 10
        return self.n_features_in_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        'Fit the model with X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data, where `n_samples` is the number of samples\n            and `n_features` is the number of features.\n\n        y : Ignored\n            Ignored.\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        self._fit(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        "Fit the model with X and apply the dimensionality reduction on X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data, where `n_samples` is the number of samples\n            and `n_features` is the number of features.\n\n        y : Ignored\n            Ignored.\n\n        Returns\n        -------\n        X_new : ndarray of shape (n_samples, n_components)\n            Transformed values.\n\n        Notes\n        -----\n        This method returns a Fortran-ordered array. To convert it to a\n        C-ordered array, use 'np.ascontiguousarray'.\n        "
        (U, S, Vt) = self._fit(X)
        U = U[:, :self.n_components_]
        if self.whiten:
            U *= sqrt(X.shape[0] - 1)
        else:
            U *= S[:self.n_components_]
        return U

    def _fit(self, X):
        if False:
            i = 10
            return i + 15
        'Dispatch to the right submethod depending on the chosen solver.'
        (xp, is_array_api_compliant) = get_namespace(X)
        if issparse(X) and self.svd_solver != 'arpack':
            raise TypeError(f'PCA only support sparse inputs with the "arpack" solver, while "{self.svd_solver}" was passed. See TruncatedSVD for a possible alternative.')
        if self.svd_solver == 'arpack' and is_array_api_compliant:
            raise ValueError("PCA with svd_solver='arpack' is not supported for Array API inputs.")
        X = self._validate_data(X, dtype=[xp.float64, xp.float32], accept_sparse=('csr', 'csc'), ensure_2d=True, copy=self.copy)
        if self.n_components is None:
            if self.svd_solver != 'arpack':
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == 'auto':
            if max(X.shape) <= 500 or n_components == 'mle':
                self._fit_svd_solver = 'full'
            elif 1 <= n_components < 0.8 * min(X.shape):
                self._fit_svd_solver = 'randomized'
            else:
                self._fit_svd_solver = 'full'
        if self._fit_svd_solver == 'full':
            return self._fit_full(X, n_components)
        elif self._fit_svd_solver in ['arpack', 'randomized']:
            return self._fit_truncated(X, n_components, self._fit_svd_solver)

    def _fit_full(self, X, n_components):
        if False:
            while True:
                i = 10
        'Fit the model by computing full SVD on X.'
        (xp, is_array_api_compliant) = get_namespace(X)
        (n_samples, n_features) = X.shape
        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported if n_samples >= n_features")
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and min(n_samples, n_features)=%r with svd_solver='full'" % (n_components, min(n_samples, n_features)))
        self.mean_ = xp.mean(X, axis=0)
        X -= self.mean_
        if not is_array_api_compliant:
            (U, S, Vt) = linalg.svd(X, full_matrices=False)
        else:
            (U, S, Vt) = xp.linalg.svd(X, full_matrices=False)
        (U, Vt) = svd_flip(U, Vt)
        components_ = Vt
        explained_variance_ = S ** 2 / (n_samples - 1)
        total_var = xp.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = xp.asarray(S, copy=True)
        if n_components == 'mle':
            n_components = _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            if is_array_api_compliant:
                explained_variance_ratio_np = _convert_to_numpy(explained_variance_ratio_, xp=xp)
            else:
                explained_variance_ratio_np = explained_variance_ratio_
            ratio_cumsum = stable_cumsum(explained_variance_ratio_np)
            n_components = np.searchsorted(ratio_cumsum, n_components, side='right') + 1
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = xp.mean(explained_variance_[n_components:])
        else:
            self.noise_variance_ = 0.0
        self.n_samples_ = n_samples
        self.components_ = components_[:n_components, :]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]
        return (U, S, Vt)

    def _fit_truncated(self, X, n_components, svd_solver):
        if False:
            while True:
                i = 10
        'Fit the model by computing truncated SVD (by ARPACK or randomized)\n        on X.\n        '
        (xp, _) = get_namespace(X)
        (n_samples, n_features) = X.shape
        if isinstance(n_components, str):
            raise ValueError("n_components=%r cannot be a string with svd_solver='%s'" % (n_components, svd_solver))
        elif not 1 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 1 and min(n_samples, n_features)=%r with svd_solver='%s'" % (n_components, min(n_samples, n_features), svd_solver))
        elif svd_solver == 'arpack' and n_components == min(n_samples, n_features):
            raise ValueError("n_components=%r must be strictly less than min(n_samples, n_features)=%r with svd_solver='%s'" % (n_components, min(n_samples, n_features), svd_solver))
        random_state = check_random_state(self.random_state)
        total_var = None
        if issparse(X):
            (self.mean_, var) = mean_variance_axis(X, axis=0)
            total_var = var.sum() * n_samples / (n_samples - 1)
            X = _implicit_column_offset(X, self.mean_)
        else:
            self.mean_ = xp.mean(X, axis=0)
            X -= self.mean_
        if svd_solver == 'arpack':
            v0 = _init_arpack_v0(min(X.shape), random_state)
            (U, S, Vt) = svds(X, k=n_components, tol=self.tol, v0=v0)
            S = S[::-1]
            (U, Vt) = svd_flip(U[:, ::-1], Vt[::-1])
        elif svd_solver == 'randomized':
            (U, S, Vt) = randomized_svd(X, n_components=n_components, n_oversamples=self.n_oversamples, n_iter=self.iterated_power, power_iteration_normalizer=self.power_iteration_normalizer, flip_sign=True, random_state=random_state)
        self.n_samples_ = n_samples
        self.components_ = Vt
        self.n_components_ = n_components
        self.explained_variance_ = S ** 2 / (n_samples - 1)
        if total_var is None:
            N = X.shape[0] - 1
            X **= 2
            total_var = xp.sum(X) / N
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self.singular_values_ = xp.asarray(S, copy=True)
        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = total_var - xp.sum(self.explained_variance_)
            self.noise_variance_ /= min(n_features, n_samples) - n_components
        else:
            self.noise_variance_ = 0.0
        return (U, S, Vt)

    def score_samples(self, X):
        if False:
            while True:
                i = 10
        'Return the log-likelihood of each sample.\n\n        See. "Pattern Recognition and Machine Learning"\n        by C. Bishop, 12.2.1 p. 574\n        or http://www.miketipping.com/papers/met-mppca.pdf\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data.\n\n        Returns\n        -------\n        ll : ndarray of shape (n_samples,)\n            Log-likelihood of each sample under the current model.\n        '
        check_is_fitted(self)
        (xp, _) = get_namespace(X)
        X = self._validate_data(X, dtype=[xp.float64, xp.float32], reset=False)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision()
        log_like = -0.5 * xp.sum(Xr * (Xr @ precision), axis=1)
        log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Return the average log-likelihood of all samples.\n\n        See. "Pattern Recognition and Machine Learning"\n        by C. Bishop, 12.2.1 p. 574\n        or http://www.miketipping.com/papers/met-mppca.pdf\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data.\n\n        y : Ignored\n            Ignored.\n\n        Returns\n        -------\n        ll : float\n            Average log-likelihood of the samples under the current model.\n        '
        (xp, _) = get_namespace(X)
        return float(xp.mean(self.score_samples(X)))

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'preserves_dtype': [np.float64, np.float32], 'array_api_support': True}