import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.stats import mvn
_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)
_doc_random_state = 'random_state : {None, int, np.random.RandomState, np.random.Generator}, optional\n    Used for drawing random variates.\n    If `seed` is `None` the `~np.random.RandomState` singleton is used.\n    If `seed` is an int, a new ``RandomState`` instance is used, seeded\n    with seed.\n    If `seed` is already a ``RandomState`` or ``Generator`` instance,\n    then that object is used.\n    Default is None.\n'

def _squeeze_output(out):
    if False:
        return 10
    '\n    Remove single-dimensional entries from array and convert to scalar,\n    if necessary.\n\n    '
    out = out.squeeze()
    if out.ndim == 0:
        out = out[()]
    return out

def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    if False:
        return 10
    '\n    Determine which eigenvalues are "small" given the spectrum.\n\n    This is for compatibility across various linear algebra functions\n    that should agree about whether or not a Hermitian matrix is numerically\n    singular and what is its numerical matrix rank.\n    This is designed to be compatible with scipy.linalg.pinvh.\n\n    Parameters\n    ----------\n    spectrum : 1d ndarray\n        Array of eigenvalues of a Hermitian matrix.\n    cond, rcond : float, optional\n        Cutoff for small eigenvalues.\n        Singular values smaller than rcond * largest_eigenvalue are\n        considered zero.\n        If None or -1, suitable machine precision is used.\n\n    Returns\n    -------\n    eps : float\n        Magnitude cutoff for numerical negligibility.\n\n    '
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1000.0, 'd': 1000000.0}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps

def _pinv_1d(v, eps=1e-05):
    if False:
        i = 10
        return i + 15
    '\n    A helper function for computing the pseudoinverse.\n\n    Parameters\n    ----------\n    v : iterable of numbers\n        This may be thought of as a vector of eigenvalues or singular values.\n    eps : float\n        Values with magnitude no greater than eps are considered negligible.\n\n    Returns\n    -------\n    v_pinv : 1d float ndarray\n        A vector of pseudo-inverted numbers.\n\n    '
    return np.array([0 if abs(x) <= eps else 1 / x for x in v], dtype=float)

class _PSD:
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.

    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().

    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)

    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().

    """

    def __init__(self, M, cond=None, rcond=None, lower=True, check_finite=True, allow_singular=True):
        if False:
            i = 10
            return i + 15
        (s, u) = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)
        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and (not allow_singular):
            raise np.linalg.LinAlgError('singular matrix')
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))
        self._pinv = None

    @property
    def pinv(self):
        if False:
            return 10
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv

class multi_rv_generic:
    """
    Class which encapsulates common functionality between all multivariate
    distributions.

    """

    def __init__(self, seed=None):
        if False:
            print('Hello World!')
        super(multi_rv_generic, self).__init__()
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        if False:
            return 10
        ' Get or set the RandomState object for generating random variates.\n\n        This can be either None, int, a RandomState instance, or a\n        np.random.Generator instance.\n\n        If None (or np.random), use the RandomState singleton used by\n        np.random.\n        If already a RandomState or Generator instance, use it.\n        If an int, use a new RandomState instance seeded with seed.\n\n        '
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        if False:
            print('Hello World!')
        self._random_state = check_random_state(seed)

    def _get_random_state(self, random_state):
        if False:
            for i in range(10):
                print('nop')
        if random_state is not None:
            return check_random_state(random_state)
        else:
            return self._random_state

class multi_rv_frozen:
    """
    Class which encapsulates common functionality between all frozen
    multivariate distributions.
    """

    @property
    def random_state(self):
        if False:
            i = 10
            return i + 15
        return self._dist._random_state

    @random_state.setter
    def random_state(self, seed):
        if False:
            while True:
                i = 10
        self._dist._random_state = check_random_state(seed)
_mvn_doc_default_callparams = 'mean : array_like, optional\n    Mean of the distribution (default zero)\ncov : array_like, optional\n    Covariance matrix of the distribution (default one)\nallow_singular : bool, optional\n    Whether to allow a singular covariance matrix.  (Default: False)\n'
_mvn_doc_callparams_note = 'Setting the parameter `mean` to `None` is equivalent to having `mean`\n    be the zero-vector. The parameter `cov` can be a scalar, in which case\n    the covariance matrix is the identity times that value, a vector of\n    diagonal entries for the covariance matrix, or a two-dimensional\n    array_like.\n    '
_mvn_doc_frozen_callparams = ''
_mvn_doc_frozen_callparams_note = 'See class definition for a detailed description of parameters.'
mvn_docdict_params = {'_mvn_doc_default_callparams': _mvn_doc_default_callparams, '_mvn_doc_callparams_note': _mvn_doc_callparams_note, '_doc_random_state': _doc_random_state}
mvn_docdict_noparams = {'_mvn_doc_default_callparams': _mvn_doc_frozen_callparams, '_mvn_doc_callparams_note': _mvn_doc_frozen_callparams_note, '_doc_random_state': _doc_random_state}

class multivariate_normal_gen(multi_rv_generic):
    """
    A multivariate normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    ``pdf(x, mean=None, cov=1, allow_singular=False)``
        Probability density function.
    ``logpdf(x, mean=None, cov=1, allow_singular=False)``
        Log of the probability density function.
    ``cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Cumulative distribution function.
    ``logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Log of the cumulative distribution function.
    ``rvs(mean=None, cov=1, size=1, random_state=None)``
        Draw random samples from a multivariate normal distribution.
    ``entropy()``
        Compute the differential entropy of the multivariate normal.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvn_doc_default_callparams)s
    %(_doc_random_state)s

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    rv = multivariate_normal(mean=None, cov=1, allow_singular=False)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    %(_mvn_doc_callparams_note)s

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \\frac{1}{\\sqrt{(2 \\pi)^k \\det \\Sigma}}
               \\exp\\left( -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu) \\right),

    where :math:`\\mu` is the mean, :math:`\\Sigma` the covariance matrix,
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_normal

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        if False:
            i = 10
            return i + 15
        super(multivariate_normal_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvn_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        if False:
            while True:
                i = 10
        '\n        Create a frozen multivariate normal distribution.\n\n        See `multivariate_normal_frozen` for more information.\n\n        '
        return multivariate_normal_frozen(mean, cov, allow_singular=allow_singular, seed=seed)

    def _process_parameters(self, dim, mean, cov):
        if False:
            return 10
        '\n        Infer dimensionality from mean or covariance matrix, ensure that\n        mean and covariance are full vector resp. matrix.\n\n        '
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        elif not np.isscalar(dim):
            raise ValueError('Dimension of random variable must be a scalar.')
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)
        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)
        if dim == 1:
            mean.shape = (1,)
            cov.shape = (1, 1)
        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." % dim)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            (rows, cols) = cov.shape
            if rows != cols:
                msg = "Array 'cov' must be square if it is two dimensional, but cov.shape = %s." % str(cov.shape)
            else:
                msg = "Dimension mismatch: array 'cov' is of shape %s, but 'mean' is a vector of length %d."
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional, but cov.ndim = %d" % cov.ndim)
        return (dim, mean, cov)

    def _process_quantiles(self, x, dim):
        if False:
            return 10
        '\n        Adjust quantiles array so that last axis labels the components of\n        each data point.\n\n        '
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _logpdf(self, x, mean, prec_U, log_det_cov, rank):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        x : ndarray\n            Points at which to evaluate the log of the probability\n            density function\n        mean : ndarray\n            Mean of the distribution\n        prec_U : ndarray\n            A decomposition such that np.dot(prec_U, prec_U.T)\n            is the precision matrix, i.e. inverse of the covariance matrix.\n        log_det_cov : float\n            Logarithm of the determinant of the covariance matrix\n        rank : int\n            Rank of the covariance matrix.\n\n        Notes\n        -----\n        As this function does no argument checking, it should not be\n        called directly; use 'logpdf' instead.\n\n        "
        dev = x - mean
        maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        if False:
            while True:
                i = 10
        '\n        Log of the multivariate normal probability density function.\n\n        Parameters\n        ----------\n        x : array_like\n            Quantiles, with the last axis of `x` denoting the components.\n        %(_mvn_doc_default_callparams)s\n\n        Returns\n        -------\n        pdf : ndarray or scalar\n            Log of the probability density function evaluated at `x`\n\n        Notes\n        -----\n        %(_mvn_doc_callparams_note)s\n\n        '
        (dim, mean, cov) = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank)
        return _squeeze_output(out)

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        if False:
            print('Hello World!')
        '\n        Multivariate normal probability density function.\n\n        Parameters\n        ----------\n        x : array_like\n            Quantiles, with the last axis of `x` denoting the components.\n        %(_mvn_doc_default_callparams)s\n\n        Returns\n        -------\n        pdf : ndarray or scalar\n            Probability density function evaluated at `x`\n\n        Notes\n        -----\n        %(_mvn_doc_callparams_note)s\n\n        '
        (dim, mean, cov) = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank))
        return _squeeze_output(out)

    def _cdf(self, x, mean, cov, maxpts, abseps, releps):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        x : ndarray\n            Points at which to evaluate the cumulative distribution function.\n        mean : ndarray\n            Mean of the distribution\n        cov : array_like\n            Covariance matrix of the distribution\n        maxpts: integer\n            The maximum number of points to use for integration\n        abseps: float\n            Absolute error tolerance\n        releps: float\n            Relative error tolerance\n\n        Notes\n        -----\n        As this function does no argument checking, it should not be\n        called directly; use 'cdf' instead.\n\n        .. versionadded:: 1.0.0\n\n        "
        lower = np.full(mean.shape, -np.inf)
        func1d = lambda x_slice: mvn.mvnun(lower, x_slice, mean, cov, maxpts, abseps, releps)[0]
        out = np.apply_along_axis(func1d, -1, x)
        return _squeeze_output(out)

    def logcdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None, abseps=1e-05, releps=1e-05):
        if False:
            for i in range(10):
                print('nop')
        '\n        Log of the multivariate normal cumulative distribution function.\n\n        Parameters\n        ----------\n        x : array_like\n            Quantiles, with the last axis of `x` denoting the components.\n        %(_mvn_doc_default_callparams)s\n        maxpts: integer, optional\n            The maximum number of points to use for integration\n            (default `1000000*dim`)\n        abseps: float, optional\n            Absolute error tolerance (default 1e-5)\n        releps: float, optional\n            Relative error tolerance (default 1e-5)\n\n        Returns\n        -------\n        cdf : ndarray or scalar\n            Log of the cumulative distribution function evaluated at `x`\n\n        Notes\n        -----\n        %(_mvn_doc_callparams_note)s\n\n        .. versionadded:: 1.0.0\n\n        '
        (dim, mean, cov) = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        _PSD(cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * dim
        out = np.log(self._cdf(x, mean, cov, maxpts, abseps, releps))
        return out

    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None, abseps=1e-05, releps=1e-05):
        if False:
            print('Hello World!')
        '\n        Multivariate normal cumulative distribution function.\n\n        Parameters\n        ----------\n        x : array_like\n            Quantiles, with the last axis of `x` denoting the components.\n        %(_mvn_doc_default_callparams)s\n        maxpts: integer, optional\n            The maximum number of points to use for integration\n            (default `1000000*dim`)\n        abseps: float, optional\n            Absolute error tolerance (default 1e-5)\n        releps: float, optional\n            Relative error tolerance (default 1e-5)\n\n        Returns\n        -------\n        cdf : ndarray or scalar\n            Cumulative distribution function evaluated at `x`\n\n        Notes\n        -----\n        %(_mvn_doc_callparams_note)s\n\n        .. versionadded:: 1.0.0\n\n        '
        (dim, mean, cov) = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        _PSD(cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * dim
        out = self._cdf(x, mean, cov, maxpts, abseps, releps)
        return out

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        if False:
            while True:
                i = 10
        '\n        Draw random samples from a multivariate normal distribution.\n\n        Parameters\n        ----------\n        %(_mvn_doc_default_callparams)s\n        size : integer, optional\n            Number of samples to draw (default 1).\n        %(_doc_random_state)s\n\n        Returns\n        -------\n        rvs : ndarray or scalar\n            Random variates of size (`size`, `N`), where `N` is the\n            dimension of the random variable.\n\n        Notes\n        -----\n        %(_mvn_doc_callparams_note)s\n\n        '
        (dim, mean, cov) = self._process_parameters(None, mean, cov)
        random_state = self._get_random_state(random_state)
        out = random_state.multivariate_normal(mean, cov, size)
        return _squeeze_output(out)

    def entropy(self, mean=None, cov=1):
        if False:
            i = 10
            return i + 15
        '\n        Compute the differential entropy of the multivariate normal.\n\n        Parameters\n        ----------\n        %(_mvn_doc_default_callparams)s\n\n        Returns\n        -------\n        h : scalar\n            Entropy of the multivariate normal distribution\n\n        Notes\n        -----\n        %(_mvn_doc_callparams_note)s\n\n        '
        (dim, mean, cov) = self._process_parameters(None, mean, cov)
        (_, logdet) = np.linalg.slogdet(2 * np.pi * np.e * cov)
        return 0.5 * logdet
multivariate_normal = multivariate_normal_gen()

class multivariate_normal_frozen(multi_rv_frozen):

    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None, maxpts=None, abseps=1e-05, releps=1e-05):
        if False:
            return 10
        '\n        Create a frozen multivariate normal distribution.\n\n        Parameters\n        ----------\n        mean : array_like, optional\n            Mean of the distribution (default zero)\n        cov : array_like, optional\n            Covariance matrix of the distribution (default one)\n        allow_singular : bool, optional\n            If this flag is True then tolerate a singular\n            covariance matrix (default False).\n        seed : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional\n            This parameter defines the object to use for drawing random\n            variates.\n            If `seed` is `None` the `~np.random.RandomState` singleton is used.\n            If `seed` is an int, a new ``RandomState`` instance is used, seeded\n            with seed.\n            If `seed` is already a ``RandomState`` or ``Generator`` instance,\n            then that object is used.\n            Default is None.\n        maxpts: integer, optional\n            The maximum number of points to use for integration of the\n            cumulative distribution function (default `1000000*dim`)\n        abseps: float, optional\n            Absolute error tolerance for the cumulative distribution function\n            (default 1e-5)\n        releps: float, optional\n            Relative error tolerance for the cumulative distribution function\n            (default 1e-5)\n\n        Examples\n        --------\n        When called with the default parameters, this will create a 1D random\n        variable with mean 0 and covariance 1:\n\n        >>> from scipy.stats import multivariate_normal\n        >>> r = multivariate_normal()\n        >>> r.mean\n        array([ 0.])\n        >>> r.cov\n        array([[1.]])\n\n        '
        self._dist = multivariate_normal_gen(seed)
        (self.dim, self.mean, self.cov) = self._dist._process_parameters(None, mean, cov)
        self.cov_info = _PSD(self.cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def logpdf(self, x):
        if False:
            while True:
                i = 10
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.mean, self.cov_info.U, self.cov_info.log_pdet, self.cov_info.rank)
        return _squeeze_output(out)

    def pdf(self, x):
        if False:
            return 10
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        if False:
            return 10
        return np.log(self.cdf(x))

    def cdf(self, x):
        if False:
            print('Hello World!')
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.mean, self.cov, self.maxpts, self.abseps, self.releps)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        if False:
            while True:
                i = 10
        return self._dist.rvs(self.mean, self.cov, size, random_state)

    def entropy(self):
        if False:
            return 10
        '\n        Computes the differential entropy of the multivariate normal.\n\n        Returns\n        -------\n        h : scalar\n            Entropy of the multivariate normal distribution\n\n        '
        log_pdet = self.cov_info.log_pdet
        rank = self.cov_info.rank
        return 0.5 * (rank * (_LOG_2PI + 1) + log_pdet)
_mvt_doc_default_callparams = '\nloc : array_like, optional\n    Location of the distribution. (default ``0``)\nshape : array_like, optional\n    Positive semidefinite matrix of the distribution. (default ``1``)\ndf : float, optional\n    Degrees of freedom of the distribution; must be greater than zero.\n    If ``np.inf`` then results are multivariate normal. The default is ``1``.\nallow_singular : bool, optional\n    Whether to allow a singular matrix. (default ``False``)\n'
_mvt_doc_callparams_note = 'Setting the parameter `loc` to ``None`` is equivalent to having `loc`\nbe the zero-vector. The parameter `shape` can be a scalar, in which case\nthe shape matrix is the identity times that value, a vector of\ndiagonal entries for the shape matrix, or a two-dimensional array_like.\n'
_mvt_doc_frozen_callparams_note = 'See class definition for a detailed description of parameters.'
mvt_docdict_params = {'_mvt_doc_default_callparams': _mvt_doc_default_callparams, '_mvt_doc_callparams_note': _mvt_doc_callparams_note, '_doc_random_state': _doc_random_state}
mvt_docdict_noparams = {'_mvt_doc_default_callparams': '', '_mvt_doc_callparams_note': _mvt_doc_frozen_callparams_note, '_doc_random_state': _doc_random_state}

class multivariate_t_gen(multi_rv_generic):
    """
    A multivariate t-distributed random variable.

    The `loc` parameter specifies the location. The `shape` parameter specifies
    the positive semidefinite shape matrix. The `df` parameter specifies the
    degrees of freedom.

    In addition to calling the methods below, the object itself may be called
    as a function to fix the location, shape matrix, and degrees of freedom
    parameters, returning a "frozen" multivariate t-distribution random.

    Methods
    -------
    ``pdf(x, loc=None, shape=1, df=1, allow_singular=False)``
        Probability density function.
    ``logpdf(x, loc=None, shape=1, df=1, allow_singular=False)``
        Log of the probability density function.
    ``rvs(loc=None, shape=1, df=1, size=1, random_state=None)``
        Draw random samples from a multivariate t-distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvt_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_mvt_doc_callparams_note)s
    The matrix `shape` must be a (symmetric) positive semidefinite matrix. The
    determinant and inverse of `shape` are computed as the pseudo-determinant
    and pseudo-inverse, respectively, so that `shape` does not need to have
    full rank.

    The probability density function for `multivariate_t` is

    .. math::

        f(x) = \\frac{\\Gamma(\\nu + p)/2}{\\Gamma(\\nu/2)\\nu^{p/2}\\pi^{p/2}|\\Sigma|^{1/2}}
               \\exp\\left[1 + \\frac{1}{\\nu} (\\mathbf{x} - \\boldsymbol{\\mu})^{\\top}
               \\boldsymbol{\\Sigma}^{-1}
               (\\mathbf{x} - \\boldsymbol{\\mu}) \\right]^{-(\\nu + p)/2},

    where :math:`p` is the dimension of :math:`\\mathbf{x}`,
    :math:`\\boldsymbol{\\mu}` is the :math:`p`-dimensional location,
    :math:`\\boldsymbol{\\Sigma}` the :math:`p \\times p`-dimensional shape
    matrix, and :math:`\\nu` is the degrees of freedom.

    .. versionadded:: 1.6.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_t
    >>> x, y = np.mgrid[-1:3:.01, -2:1.5:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_t([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]], df=2)
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.set_aspect('equal')
    >>> plt.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        if False:
            print('Hello World!')
        '\n        Initialize a multivariate t-distributed random variable.\n\n        Parameters\n        ----------\n        seed : Random state.\n\n        '
        super(multivariate_t_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvt_docdict_params)
        self._random_state = check_random_state(seed)

    def __call__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        if False:
            while True:
                i = 10
        '\n        Create a frozen multivariate t-distribution. See\n        `multivariate_t_frozen` for parameters.\n\n        '
        if df == np.inf:
            return multivariate_normal_frozen(mean=loc, cov=shape, allow_singular=allow_singular, seed=seed)
        return multivariate_t_frozen(loc=loc, shape=shape, df=df, allow_singular=allow_singular, seed=seed)

    def pdf(self, x, loc=None, shape=1, df=1, allow_singular=False):
        if False:
            i = 10
            return i + 15
        '\n        Multivariate t-distribution probability density function.\n\n        Parameters\n        ----------\n        x : array_like\n            Points at which to evaluate the probability density function.\n        %(_mvt_doc_default_callparams)s\n\n        Returns\n        -------\n        pdf : Probability density function evaluated at `x`.\n\n        Examples\n        --------\n        >>> from scipy.stats import multivariate_t\n        >>> x = [0.4, 5]\n        >>> loc = [0, 1]\n        >>> shape = [[1, 0.1], [0.1, 1]]\n        >>> df = 7\n        >>> multivariate_t.pdf(x, loc, shape, df)\n        array([0.00075713])\n\n        '
        (dim, loc, shape, df) = self._process_parameters(loc, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape, allow_singular=allow_singular)
        logpdf = self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df, dim, shape_info.rank)
        return np.exp(logpdf)

    def logpdf(self, x, loc=None, shape=1, df=1):
        if False:
            print('Hello World!')
        '\n        Log of the multivariate t-distribution probability density function.\n\n        Parameters\n        ----------\n        x : array_like\n            Points at which to evaluate the log of the probability density\n            function.\n        %(_mvt_doc_default_callparams)s\n\n        Returns\n        -------\n        logpdf : Log of the probability density function evaluated at `x`.\n\n        Examples\n        --------\n        >>> from scipy.stats import multivariate_t\n        >>> x = [0.4, 5]\n        >>> loc = [0, 1]\n        >>> shape = [[1, 0.1], [0.1, 1]]\n        >>> df = 7\n        >>> multivariate_t.logpdf(x, loc, shape, df)\n        array([-7.1859802])\n\n        See Also\n        --------\n        pdf : Probability density function.\n\n        '
        (dim, loc, shape, df) = self._process_parameters(loc, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape)
        return self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df, dim, shape_info.rank)

    def _logpdf(self, x, loc, prec_U, log_pdet, df, dim, rank):
        if False:
            for i in range(10):
                print('nop')
        "Utility method `pdf`, `logpdf` for parameters.\n\n        Parameters\n        ----------\n        x : ndarray\n            Points at which to evaluate the log of the probability density\n            function.\n        loc : ndarray\n            Location of the distribution.\n        prec_U : ndarray\n            A decomposition such that `np.dot(prec_U, prec_U.T)` is the inverse\n            of the shape matrix.\n        log_pdet : float\n            Logarithm of the determinant of the shape matrix.\n        df : float\n            Degrees of freedom of the distribution.\n        dim : int\n            Dimension of the quantiles x.\n        rank : int\n            Rank of the shape matrix.\n\n        Notes\n        -----\n        As this function does no argument checking, it should not be called\n        directly; use 'logpdf' instead.\n\n        "
        if df == np.inf:
            return multivariate_normal._logpdf(x, loc, prec_U, log_pdet, rank)
        dev = x - loc
        maha = np.square(np.dot(dev, prec_U)).sum(axis=-1)
        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim / 2.0 * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + 1.0 / df * maha)
        return _squeeze_output(A - B - C - D + E)

    def rvs(self, loc=None, shape=1, df=1, size=1, random_state=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draw random samples from a multivariate t-distribution.\n\n        Parameters\n        ----------\n        %(_mvt_doc_default_callparams)s\n        size : integer, optional\n            Number of samples to draw (default 1).\n        %(_doc_random_state)s\n\n        Returns\n        -------\n        rvs : ndarray or scalar\n            Random variates of size (`size`, `P`), where `P` is the\n            dimension of the random variable.\n\n        Examples\n        --------\n        >>> from scipy.stats import multivariate_t\n        >>> x = [0.4, 5]\n        >>> loc = [0, 1]\n        >>> shape = [[1, 0.1], [0.1, 1]]\n        >>> df = 7\n        >>> multivariate_t.rvs(loc, shape, df)\n        array([[0.93477495, 3.00408716]])\n\n        '
        (dim, loc, shape, df) = self._process_parameters(loc, shape, df)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state
        if np.isinf(df):
            x = np.ones(size)
        else:
            x = rng.chisquare(df, size=size) / df
        z = rng.multivariate_normal(np.zeros(dim), shape, size=size)
        samples = loc + z / np.sqrt(x)[:, None]
        return _squeeze_output(samples)

    def _process_quantiles(self, x, dim):
        if False:
            while True:
                i = 10
        '\n        Adjust quantiles array so that last axis labels the components of\n        each data point.\n\n        '
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _process_parameters(self, loc, shape, df):
        if False:
            for i in range(10):
                print('nop')
        '\n        Infer dimensionality from location array and shape matrix, handle\n        defaults, and ensure compatible dimensions.\n\n        '
        if loc is None and shape is None:
            loc = np.asarray(0, dtype=float)
            shape = np.asarray(1, dtype=float)
            dim = 1
        elif loc is None:
            shape = np.asarray(shape, dtype=float)
            if shape.ndim < 2:
                dim = 1
            else:
                dim = shape.shape[0]
            loc = np.zeros(dim)
        elif shape is None:
            loc = np.asarray(loc, dtype=float)
            dim = loc.size
            shape = np.eye(dim)
        else:
            shape = np.asarray(shape, dtype=float)
            loc = np.asarray(loc, dtype=float)
            dim = loc.size
        if dim == 1:
            loc.shape = (1,)
            shape.shape = (1, 1)
        if loc.ndim != 1 or loc.shape[0] != dim:
            raise ValueError("Array 'loc' must be a vector of length %d." % dim)
        if shape.ndim == 0:
            shape = shape * np.eye(dim)
        elif shape.ndim == 1:
            shape = np.diag(shape)
        elif shape.ndim == 2 and shape.shape != (dim, dim):
            (rows, cols) = shape.shape
            if rows != cols:
                msg = "Array 'cov' must be square if it is two dimensional, but cov.shape = %s." % str(shape.shape)
            else:
                msg = "Dimension mismatch: array 'cov' is of shape %s, but 'loc' is a vector of length %d."
                msg = msg % (str(shape.shape), len(loc))
            raise ValueError(msg)
        elif shape.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional, but cov.ndim = %d" % shape.ndim)
        if df is None:
            df = 1
        elif df <= 0:
            raise ValueError("'df' must be greater than zero.")
        elif np.isnan(df):
            raise ValueError("'df' is 'nan' but must be greater than zero or 'np.inf'.")
        return (dim, loc, shape, df)

class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a frozen multivariate t distribution.\n\n        Parameters\n        ----------\n        %(_mvt_doc_default_callparams)s\n\n        Examples\n        --------\n        >>> loc = np.zeros(3)\n        >>> shape = np.eye(3)\n        >>> df = 10\n        >>> dist = multivariate_t(loc, shape, df)\n        >>> dist.rvs()\n        array([[ 0.81412036, -1.53612361,  0.42199647]])\n        >>> dist.pdf([1, 1, 1])\n        array([0.01237803])\n\n        '
        self._dist = multivariate_t_gen(seed)
        (dim, loc, shape, df) = self._dist._process_parameters(loc, shape, df)
        (self.dim, self.loc, self.shape, self.df) = (dim, loc, shape, df)
        self.shape_info = _PSD(shape, allow_singular=allow_singular)

    def logpdf(self, x):
        if False:
            print('Hello World!')
        x = self._dist._process_quantiles(x, self.dim)
        U = self.shape_info.U
        log_pdet = self.shape_info.log_pdet
        return self._dist._logpdf(x, self.loc, U, log_pdet, self.df, self.dim, self.shape_info.rank)

    def pdf(self, x):
        if False:
            print('Hello World!')
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        if False:
            print('Hello World!')
        return self._dist.rvs(loc=self.loc, shape=self.shape, df=self.df, size=size, random_state=random_state)
multivariate_t = multivariate_t_gen()
for name in ['logpdf', 'pdf', 'rvs']:
    method = multivariate_t_gen.__dict__[name]
    method_frozen = multivariate_t_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_params)