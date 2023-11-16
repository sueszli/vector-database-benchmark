from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
__all__ = ['Covariance']

class Covariance:
    """
    Representation of a covariance matrix

    Calculations involving covariance matrices (e.g. data whitening,
    multivariate normal function evaluation) are often performed more
    efficiently using a decomposition of the covariance matrix instead of the
    covariance matrix itself. This class allows the user to construct an
    object representing a covariance matrix using any of several
    decompositions and perform calculations using a common interface.

    .. note::

        The `Covariance` class cannot be instantiated directly. Instead, use
        one of the factory methods (e.g. `Covariance.from_diagonal`).

    Examples
    --------
    The `Covariance` class is is used by calling one of its
    factory methods to create a `Covariance` object, then pass that
    representation of the `Covariance` matrix as a shape parameter of a
    multivariate distribution.

    For instance, the multivariate normal distribution can accept an array
    representing a covariance matrix:

    >>> from scipy import stats
    >>> import numpy as np
    >>> d = [1, 2, 3]
    >>> A = np.diag(d)  # a diagonal covariance matrix
    >>> x = [4, -2, 5]  # a point of interest
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=A)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    but the calculations are performed in a very generic way that does not
    take advantage of any special properties of the covariance matrix. Because
    our covariance matrix is diagonal, we can use ``Covariance.from_diagonal``
    to create an object representing the covariance matrix, and
    `multivariate_normal` can use this to compute the probability density
    function more efficiently.

    >>> cov = stats.Covariance.from_diagonal(d)
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=cov)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    """

    def __init__(self):
        if False:
            print('Hello World!')
        message = 'The `Covariance` class cannot be instantiated directly. Please use one of the factory methods (e.g. `Covariance.from_diagonal`).'
        raise NotImplementedError(message)

    @staticmethod
    def from_diagonal(diagonal):
        if False:
            print('Hello World!')
        '\n        Return a representation of a covariance matrix from its diagonal.\n\n        Parameters\n        ----------\n        diagonal : array_like\n            The diagonal elements of a diagonal matrix.\n\n        Notes\n        -----\n        Let the diagonal elements of a diagonal covariance matrix :math:`D` be\n        stored in the vector :math:`d`.\n\n        When all elements of :math:`d` are strictly positive, whitening of a\n        data point :math:`x` is performed by computing\n        :math:`x \\cdot d^{-1/2}`, where the inverse square root can be taken\n        element-wise.\n        :math:`\\log\\det{D}` is calculated as :math:`-2 \\sum(\\log{d})`,\n        where the :math:`\\log` operation is performed element-wise.\n\n        This `Covariance` class supports singular covariance matrices. When\n        computing ``_log_pdet``, non-positive elements of :math:`d` are\n        ignored. Whitening is not well defined when the point to be whitened\n        does not lie in the span of the columns of the covariance matrix. The\n        convention taken here is to treat the inverse square root of\n        non-positive elements of :math:`d` as zeros.\n\n        Examples\n        --------\n        Prepare a symmetric positive definite covariance matrix ``A`` and a\n        data point ``x``.\n\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> rng = np.random.default_rng()\n        >>> n = 5\n        >>> A = np.diag(rng.random(n))\n        >>> x = rng.random(size=n)\n\n        Extract the diagonal from ``A`` and create the `Covariance` object.\n\n        >>> d = np.diag(A)\n        >>> cov = stats.Covariance.from_diagonal(d)\n\n        Compare the functionality of the `Covariance` object against a\n        reference implementations.\n\n        >>> res = cov.whiten(x)\n        >>> ref = np.diag(d**-0.5) @ x\n        >>> np.allclose(res, ref)\n        True\n        >>> res = cov.log_pdet\n        >>> ref = np.linalg.slogdet(A)[-1]\n        >>> np.allclose(res, ref)\n        True\n\n        '
        return CovViaDiagonal(diagonal)

    @staticmethod
    def from_precision(precision, covariance=None):
        if False:
            while True:
                i = 10
        '\n        Return a representation of a covariance from its precision matrix.\n\n        Parameters\n        ----------\n        precision : array_like\n            The precision matrix; that is, the inverse of a square, symmetric,\n            positive definite covariance matrix.\n        covariance : array_like, optional\n            The square, symmetric, positive definite covariance matrix. If not\n            provided, this may need to be calculated (e.g. to evaluate the\n            cumulative distribution function of\n            `scipy.stats.multivariate_normal`) by inverting `precision`.\n\n        Notes\n        -----\n        Let the covariance matrix be :math:`A`, its precision matrix be\n        :math:`P = A^{-1}`, and :math:`L` be the lower Cholesky factor such\n        that :math:`L L^T = P`.\n        Whitening of a data point :math:`x` is performed by computing\n        :math:`x^T L`. :math:`\\log\\det{A}` is calculated as\n        :math:`-2tr(\\log{L})`, where the :math:`\\log` operation is performed\n        element-wise.\n\n        This `Covariance` class does not support singular covariance matrices\n        because the precision matrix does not exist for a singular covariance\n        matrix.\n\n        Examples\n        --------\n        Prepare a symmetric positive definite precision matrix ``P`` and a\n        data point ``x``. (If the precision matrix is not already available,\n        consider the other factory methods of the `Covariance` class.)\n\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> rng = np.random.default_rng()\n        >>> n = 5\n        >>> P = rng.random(size=(n, n))\n        >>> P = P @ P.T  # a precision matrix must be positive definite\n        >>> x = rng.random(size=n)\n\n        Create the `Covariance` object.\n\n        >>> cov = stats.Covariance.from_precision(P)\n\n        Compare the functionality of the `Covariance` object against\n        reference implementations.\n\n        >>> res = cov.whiten(x)\n        >>> ref = x @ np.linalg.cholesky(P)\n        >>> np.allclose(res, ref)\n        True\n        >>> res = cov.log_pdet\n        >>> ref = -np.linalg.slogdet(P)[-1]\n        >>> np.allclose(res, ref)\n        True\n\n        '
        return CovViaPrecision(precision, covariance)

    @staticmethod
    def from_cholesky(cholesky):
        if False:
            for i in range(10):
                print('nop')
        '\n        Representation of a covariance provided via the (lower) Cholesky factor\n\n        Parameters\n        ----------\n        cholesky : array_like\n            The lower triangular Cholesky factor of the covariance matrix.\n\n        Notes\n        -----\n        Let the covariance matrix be :math:`A` and :math:`L` be the lower\n        Cholesky factor such that :math:`L L^T = A`.\n        Whitening of a data point :math:`x` is performed by computing\n        :math:`L^{-1} x`. :math:`\\log\\det{A}` is calculated as\n        :math:`2tr(\\log{L})`, where the :math:`\\log` operation is performed\n        element-wise.\n\n        This `Covariance` class does not support singular covariance matrices\n        because the Cholesky decomposition does not exist for a singular\n        covariance matrix.\n\n        Examples\n        --------\n        Prepare a symmetric positive definite covariance matrix ``A`` and a\n        data point ``x``.\n\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> rng = np.random.default_rng()\n        >>> n = 5\n        >>> A = rng.random(size=(n, n))\n        >>> A = A @ A.T  # make the covariance symmetric positive definite\n        >>> x = rng.random(size=n)\n\n        Perform the Cholesky decomposition of ``A`` and create the\n        `Covariance` object.\n\n        >>> L = np.linalg.cholesky(A)\n        >>> cov = stats.Covariance.from_cholesky(L)\n\n        Compare the functionality of the `Covariance` object against\n        reference implementation.\n\n        >>> from scipy.linalg import solve_triangular\n        >>> res = cov.whiten(x)\n        >>> ref = solve_triangular(L, x, lower=True)\n        >>> np.allclose(res, ref)\n        True\n        >>> res = cov.log_pdet\n        >>> ref = np.linalg.slogdet(A)[-1]\n        >>> np.allclose(res, ref)\n        True\n\n        '
        return CovViaCholesky(cholesky)

    @staticmethod
    def from_eigendecomposition(eigendecomposition):
        if False:
            while True:
                i = 10
        '\n        Representation of a covariance provided via eigendecomposition\n\n        Parameters\n        ----------\n        eigendecomposition : sequence\n            A sequence (nominally a tuple) containing the eigenvalue and\n            eigenvector arrays as computed by `scipy.linalg.eigh` or\n            `numpy.linalg.eigh`.\n\n        Notes\n        -----\n        Let the covariance matrix be :math:`A`, let :math:`V` be matrix of\n        eigenvectors, and let :math:`W` be the diagonal matrix of eigenvalues\n        such that `V W V^T = A`.\n\n        When all of the eigenvalues are strictly positive, whitening of a\n        data point :math:`x` is performed by computing\n        :math:`x^T (V W^{-1/2})`, where the inverse square root can be taken\n        element-wise.\n        :math:`\\log\\det{A}` is calculated as  :math:`tr(\\log{W})`,\n        where the :math:`\\log` operation is performed element-wise.\n\n        This `Covariance` class supports singular covariance matrices. When\n        computing ``_log_pdet``, non-positive eigenvalues are ignored.\n        Whitening is not well defined when the point to be whitened\n        does not lie in the span of the columns of the covariance matrix. The\n        convention taken here is to treat the inverse square root of\n        non-positive eigenvalues as zeros.\n\n        Examples\n        --------\n        Prepare a symmetric positive definite covariance matrix ``A`` and a\n        data point ``x``.\n\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> rng = np.random.default_rng()\n        >>> n = 5\n        >>> A = rng.random(size=(n, n))\n        >>> A = A @ A.T  # make the covariance symmetric positive definite\n        >>> x = rng.random(size=n)\n\n        Perform the eigendecomposition of ``A`` and create the `Covariance`\n        object.\n\n        >>> w, v = np.linalg.eigh(A)\n        >>> cov = stats.Covariance.from_eigendecomposition((w, v))\n\n        Compare the functionality of the `Covariance` object against\n        reference implementations.\n\n        >>> res = cov.whiten(x)\n        >>> ref = x @ (v @ np.diag(w**-0.5))\n        >>> np.allclose(res, ref)\n        True\n        >>> res = cov.log_pdet\n        >>> ref = np.linalg.slogdet(A)[-1]\n        >>> np.allclose(res, ref)\n        True\n\n        '
        return CovViaEigendecomposition(eigendecomposition)

    def whiten(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform a whitening transformation on data.\n\n        "Whitening" ("white" as in "white noise", in which each frequency has\n        equal magnitude) transforms a set of random variables into a new set of\n        random variables with unit-diagonal covariance. When a whitening\n        transform is applied to a sample of points distributed according to\n        a multivariate normal distribution with zero mean, the covariance of\n        the transformed sample is approximately the identity matrix.\n\n        Parameters\n        ----------\n        x : array_like\n            An array of points. The last dimension must correspond with the\n            dimensionality of the space, i.e., the number of columns in the\n            covariance matrix.\n\n        Returns\n        -------\n        x_ : array_like\n            The transformed array of points.\n\n        References\n        ----------\n        .. [1] "Whitening Transformation". Wikipedia.\n               https://en.wikipedia.org/wiki/Whitening_transformation\n        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of\n               coloring linear transformation". Transactions of VSB 18.2\n               (2018): 31-35. :doi:`10.31490/tces-2018-0013`\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> rng = np.random.default_rng()\n        >>> n = 3\n        >>> A = rng.random(size=(n, n))\n        >>> cov_array = A @ A.T  # make matrix symmetric positive definite\n        >>> precision = np.linalg.inv(cov_array)\n        >>> cov_object = stats.Covariance.from_precision(precision)\n        >>> x = rng.multivariate_normal(np.zeros(n), cov_array, size=(10000))\n        >>> x_ = cov_object.whiten(x)\n        >>> np.cov(x_, rowvar=False)  # near-identity covariance\n        array([[0.97862122, 0.00893147, 0.02430451],\n               [0.00893147, 0.96719062, 0.02201312],\n               [0.02430451, 0.02201312, 0.99206881]])\n\n        '
        return self._whiten(np.asarray(x))

    def colorize(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform a colorizing transformation on data.\n\n        "Colorizing" ("color" as in "colored noise", in which different\n        frequencies may have different magnitudes) transforms a set of\n        uncorrelated random variables into a new set of random variables with\n        the desired covariance. When a coloring transform is applied to a\n        sample of points distributed according to a multivariate normal\n        distribution with identity covariance and zero mean, the covariance of\n        the transformed sample is approximately the covariance matrix used\n        in the coloring transform.\n\n        Parameters\n        ----------\n        x : array_like\n            An array of points. The last dimension must correspond with the\n            dimensionality of the space, i.e., the number of columns in the\n            covariance matrix.\n\n        Returns\n        -------\n        x_ : array_like\n            The transformed array of points.\n\n        References\n        ----------\n        .. [1] "Whitening Transformation". Wikipedia.\n               https://en.wikipedia.org/wiki/Whitening_transformation\n        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of\n               coloring linear transformation". Transactions of VSB 18.2\n               (2018): 31-35. :doi:`10.31490/tces-2018-0013`\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> rng = np.random.default_rng(1638083107694713882823079058616272161)\n        >>> n = 3\n        >>> A = rng.random(size=(n, n))\n        >>> cov_array = A @ A.T  # make matrix symmetric positive definite\n        >>> cholesky = np.linalg.cholesky(cov_array)\n        >>> cov_object = stats.Covariance.from_cholesky(cholesky)\n        >>> x = rng.multivariate_normal(np.zeros(n), np.eye(n), size=(10000))\n        >>> x_ = cov_object.colorize(x)\n        >>> cov_data = np.cov(x_, rowvar=False)\n        >>> np.allclose(cov_data, cov_array, rtol=3e-2)\n        True\n        '
        return self._colorize(np.asarray(x))

    @property
    def log_pdet(self):
        if False:
            print('Hello World!')
        '\n        Log of the pseudo-determinant of the covariance matrix\n        '
        return np.array(self._log_pdet, dtype=float)[()]

    @property
    def rank(self):
        if False:
            i = 10
            return i + 15
        '\n        Rank of the covariance matrix\n        '
        return np.array(self._rank, dtype=int)[()]

    @property
    def covariance(self):
        if False:
            return 10
        '\n        Explicit representation of the covariance matrix\n        '
        return self._covariance

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shape of the covariance array\n        '
        return self._shape

    def _validate_matrix(self, A, name):
        if False:
            print('Hello World!')
        A = np.atleast_2d(A)
        (m, n) = A.shape[-2:]
        if m != n or A.ndim != 2 or (not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating))):
            message = f'The input `{name}` must be a square, two-dimensional array of real numbers.'
            raise ValueError(message)
        return A

    def _validate_vector(self, A, name):
        if False:
            for i in range(10):
                print('nop')
        A = np.atleast_1d(A)
        if A.ndim != 1 or not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
            message = f'The input `{name}` must be a one-dimensional array of real numbers.'
            raise ValueError(message)
        return A

class CovViaPrecision(Covariance):

    def __init__(self, precision, covariance=None):
        if False:
            return 10
        precision = self._validate_matrix(precision, 'precision')
        if covariance is not None:
            covariance = self._validate_matrix(covariance, 'covariance')
            message = '`precision.shape` must equal `covariance.shape`.'
            if precision.shape != covariance.shape:
                raise ValueError(message)
        self._chol_P = np.linalg.cholesky(precision)
        self._log_pdet = -2 * np.log(np.diag(self._chol_P)).sum(axis=-1)
        self._rank = precision.shape[-1]
        self._precision = precision
        self._cov_matrix = covariance
        self._shape = precision.shape
        self._allow_singular = False

    def _whiten(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x @ self._chol_P

    @cached_property
    def _covariance(self):
        if False:
            return 10
        n = self._shape[-1]
        return linalg.cho_solve((self._chol_P, True), np.eye(n)) if self._cov_matrix is None else self._cov_matrix

    def _colorize(self, x):
        if False:
            while True:
                i = 10
        return linalg.solve_triangular(self._chol_P.T, x.T, lower=False).T

def _dot_diag(x, d):
    if False:
        for i in range(10):
            print('nop')
    return x * d if x.ndim < 2 else x * np.expand_dims(d, -2)

class CovViaDiagonal(Covariance):

    def __init__(self, diagonal):
        if False:
            for i in range(10):
                print('nop')
        diagonal = self._validate_vector(diagonal, 'diagonal')
        i_zero = diagonal <= 0
        positive_diagonal = np.array(diagonal, dtype=np.float64)
        positive_diagonal[i_zero] = 1
        self._log_pdet = np.sum(np.log(positive_diagonal), axis=-1)
        psuedo_reciprocals = 1 / np.sqrt(positive_diagonal)
        psuedo_reciprocals[i_zero] = 0
        self._sqrt_diagonal = np.sqrt(diagonal)
        self._LP = psuedo_reciprocals
        self._rank = positive_diagonal.shape[-1] - i_zero.sum(axis=-1)
        self._covariance = np.apply_along_axis(np.diag, -1, diagonal)
        self._i_zero = i_zero
        self._shape = self._covariance.shape
        self._allow_singular = True

    def _whiten(self, x):
        if False:
            for i in range(10):
                print('nop')
        return _dot_diag(x, self._LP)

    def _colorize(self, x):
        if False:
            return 10
        return _dot_diag(x, self._sqrt_diagonal)

    def _support_mask(self, x):
        if False:
            return 10
        '\n        Check whether x lies in the support of the distribution.\n        '
        return ~np.any(_dot_diag(x, self._i_zero), axis=-1)

class CovViaCholesky(Covariance):

    def __init__(self, cholesky):
        if False:
            while True:
                i = 10
        L = self._validate_matrix(cholesky, 'cholesky')
        self._factor = L
        self._log_pdet = 2 * np.log(np.diag(self._factor)).sum(axis=-1)
        self._rank = L.shape[-1]
        self._shape = L.shape
        self._allow_singular = False

    @cached_property
    def _covariance(self):
        if False:
            print('Hello World!')
        return self._factor @ self._factor.T

    def _whiten(self, x):
        if False:
            return 10
        res = linalg.solve_triangular(self._factor, x.T, lower=True).T
        return res

    def _colorize(self, x):
        if False:
            i = 10
            return i + 15
        return x @ self._factor.T

class CovViaEigendecomposition(Covariance):

    def __init__(self, eigendecomposition):
        if False:
            while True:
                i = 10
        (eigenvalues, eigenvectors) = eigendecomposition
        eigenvalues = self._validate_vector(eigenvalues, 'eigenvalues')
        eigenvectors = self._validate_matrix(eigenvectors, 'eigenvectors')
        message = 'The shapes of `eigenvalues` and `eigenvectors` must be compatible.'
        try:
            eigenvalues = np.expand_dims(eigenvalues, -2)
            (eigenvectors, eigenvalues) = np.broadcast_arrays(eigenvectors, eigenvalues)
            eigenvalues = eigenvalues[..., 0, :]
        except ValueError:
            raise ValueError(message)
        i_zero = eigenvalues <= 0
        positive_eigenvalues = np.array(eigenvalues, dtype=np.float64)
        positive_eigenvalues[i_zero] = 1
        self._log_pdet = np.sum(np.log(positive_eigenvalues), axis=-1)
        psuedo_reciprocals = 1 / np.sqrt(positive_eigenvalues)
        psuedo_reciprocals[i_zero] = 0
        self._LP = eigenvectors * psuedo_reciprocals
        self._LA = eigenvectors * np.sqrt(eigenvalues)
        self._rank = positive_eigenvalues.shape[-1] - i_zero.sum(axis=-1)
        self._w = eigenvalues
        self._v = eigenvectors
        self._shape = eigenvectors.shape
        self._null_basis = eigenvectors * i_zero
        self._eps = _multivariate._eigvalsh_to_eps(eigenvalues) * 10 ** 3
        self._allow_singular = True

    def _whiten(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x @ self._LP

    def _colorize(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x @ self._LA.T

    @cached_property
    def _covariance(self):
        if False:
            for i in range(10):
                print('nop')
        return self._v * self._w @ self._v.T

    def _support_mask(self, x):
        if False:
            return 10
        '\n        Check whether x lies in the support of the distribution.\n        '
        residual = np.linalg.norm(x @ self._null_basis, axis=-1)
        in_support = residual < self._eps
        return in_support

class CovViaPSD(Covariance):
    """
    Representation of a covariance provided via an instance of _PSD
    """

    def __init__(self, psd):
        if False:
            return 10
        self._LP = psd.U
        self._log_pdet = psd.log_pdet
        self._rank = psd.rank
        self._covariance = psd._M
        self._shape = psd._M.shape
        self._psd = psd
        self._allow_singular = False

    def _whiten(self, x):
        if False:
            print('Hello World!')
        return x @ self._LP

    def _support_mask(self, x):
        if False:
            return 10
        return self._psd._support_mask(x)