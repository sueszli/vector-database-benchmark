"""

Created on Fri Aug 17 13:10:52 2012

Author: Josef Perktold
License: BSD-3
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc

def clip_evals(x, value=0):
    if False:
        print('Hello World!')
    (evals, evecs) = np.linalg.eigh(x)
    clipped = np.any(evals < value)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return (x_new, clipped)

def corr_nearest(corr, threshold=1e-15, n_fact=100):
    if False:
        return 10
    '\n    Find the nearest correlation matrix that is positive semi-definite.\n\n    The function iteratively adjust the correlation matrix by clipping the\n    eigenvalues of a difference matrix. The diagonal elements are set to one.\n\n    Parameters\n    ----------\n    corr : ndarray, (k, k)\n        initial correlation matrix\n    threshold : float\n        clipping threshold for smallest eigenvalue, see Notes\n    n_fact : int or float\n        factor to determine the maximum number of iterations. The maximum\n        number of iterations is the integer part of the number of columns in\n        the correlation matrix times n_fact.\n\n    Returns\n    -------\n    corr_new : ndarray, (optional)\n        corrected correlation matrix\n\n    Notes\n    -----\n    The smallest eigenvalue of the corrected correlation matrix is\n    approximately equal to the ``threshold``.\n    If the threshold=0, then the smallest eigenvalue of the correlation matrix\n    might be negative, but zero within a numerical error, for example in the\n    range of -1e-16.\n\n    Assumes input correlation matrix is symmetric.\n\n    Stops after the first step if correlation matrix is already positive\n    semi-definite or positive definite, so that smallest eigenvalue is above\n    threshold. In this case, the returned array is not the original, but\n    is equal to it within numerical precision.\n\n    See Also\n    --------\n    corr_clipped\n    cov_nearest\n\n    '
    k_vars = corr.shape[0]
    if k_vars != corr.shape[1]:
        raise ValueError('matrix is not square')
    diff = np.zeros(corr.shape)
    x_new = corr.copy()
    diag_idx = np.arange(k_vars)
    for ii in range(int(len(corr) * n_fact)):
        x_adj = x_new - diff
        (x_psd, clipped) = clip_evals(x_adj, value=threshold)
        if not clipped:
            x_new = x_psd
            break
        diff = x_psd - x_adj
        x_new = x_psd.copy()
        x_new[diag_idx, diag_idx] = 1
    else:
        warnings.warn(iteration_limit_doc, IterationLimitWarning)
    return x_new

def corr_clipped(corr, threshold=1e-15):
    if False:
        print('Hello World!')
    '\n    Find a near correlation matrix that is positive semi-definite\n\n    This function clips the eigenvalues, replacing eigenvalues smaller than\n    the threshold by the threshold. The new matrix is normalized, so that the\n    diagonal elements are one.\n    Compared to corr_nearest, the distance between the original correlation\n    matrix and the positive definite correlation matrix is larger, however,\n    it is much faster since it only computes eigenvalues once.\n\n    Parameters\n    ----------\n    corr : ndarray, (k, k)\n        initial correlation matrix\n    threshold : float\n        clipping threshold for smallest eigenvalue, see Notes\n\n    Returns\n    -------\n    corr_new : ndarray, (optional)\n        corrected correlation matrix\n\n\n    Notes\n    -----\n    The smallest eigenvalue of the corrected correlation matrix is\n    approximately equal to the ``threshold``. In examples, the\n    smallest eigenvalue can be by a factor of 10 smaller than the threshold,\n    e.g. threshold 1e-8 can result in smallest eigenvalue in the range\n    between 1e-9 and 1e-8.\n    If the threshold=0, then the smallest eigenvalue of the correlation matrix\n    might be negative, but zero within a numerical error, for example in the\n    range of -1e-16.\n\n    Assumes input correlation matrix is symmetric. The diagonal elements of\n    returned correlation matrix is set to ones.\n\n    If the correlation matrix is already positive semi-definite given the\n    threshold, then the original correlation matrix is returned.\n\n    ``cov_clipped`` is 40 or more times faster than ``cov_nearest`` in simple\n    example, but has a slightly larger approximation error.\n\n    See Also\n    --------\n    corr_nearest\n    cov_nearest\n\n    '
    (x_new, clipped) = clip_evals(corr, value=threshold)
    if not clipped:
        return corr
    x_std = np.sqrt(np.diag(x_new))
    x_new = x_new / x_std / x_std[:, None]
    return x_new

def cov_nearest(cov, method='clipped', threshold=1e-15, n_fact=100, return_all=False):
    if False:
        print('Hello World!')
    '\n    Find the nearest covariance matrix that is positive (semi-) definite\n\n    This leaves the diagonal, i.e. the variance, unchanged\n\n    Parameters\n    ----------\n    cov : ndarray, (k,k)\n        initial covariance matrix\n    method : str\n        if "clipped", then the faster but less accurate ``corr_clipped`` is\n        used.if "nearest", then ``corr_nearest`` is used\n    threshold : float\n        clipping threshold for smallest eigen value, see Notes\n    n_fact : int or float\n        factor to determine the maximum number of iterations in\n        ``corr_nearest``. See its doc string\n    return_all : bool\n        if False (default), then only the covariance matrix is returned.\n        If True, then correlation matrix and standard deviation are\n        additionally returned.\n\n    Returns\n    -------\n    cov_ : ndarray\n        corrected covariance matrix\n    corr_ : ndarray, (optional)\n        corrected correlation matrix\n    std_ : ndarray, (optional)\n        standard deviation\n\n\n    Notes\n    -----\n    This converts the covariance matrix to a correlation matrix. Then, finds\n    the nearest correlation matrix that is positive semidefinite and converts\n    it back to a covariance matrix using the initial standard deviation.\n\n    The smallest eigenvalue of the intermediate correlation matrix is\n    approximately equal to the ``threshold``.\n    If the threshold=0, then the smallest eigenvalue of the correlation matrix\n    might be negative, but zero within a numerical error, for example in the\n    range of -1e-16.\n\n    Assumes input covariance matrix is symmetric.\n\n    See Also\n    --------\n    corr_nearest\n    corr_clipped\n    '
    from statsmodels.stats.moment_helpers import cov2corr, corr2cov
    (cov_, std_) = cov2corr(cov, return_std=True)
    if method == 'clipped':
        corr_ = corr_clipped(cov_, threshold=threshold)
    else:
        corr_ = corr_nearest(cov_, threshold=threshold, n_fact=n_fact)
    cov_ = corr2cov(corr_, std_)
    if return_all:
        return (cov_, corr_, std_)
    else:
        return cov_

def _nmono_linesearch(obj, grad, x, d, obj_hist, M=10, sig1=0.1, sig2=0.9, gam=0.0001, maxiter=100):
    if False:
        print('Hello World!')
    "\n    Implements the non-monotone line search of Grippo et al. (1986),\n    as described in Birgin, Martinez and Raydan (2013).\n\n    Parameters\n    ----------\n    obj : real-valued function\n        The objective function, to be minimized\n    grad : vector-valued function\n        The gradient of the objective function\n    x : array_like\n        The starting point for the line search\n    d : array_like\n        The search direction\n    obj_hist : array_like\n        Objective function history (must contain at least one value)\n    M : positive int\n        Number of previous function points to consider (see references\n        for details).\n    sig1 : real\n        Tuning parameter, see references for details.\n    sig2 : real\n        Tuning parameter, see references for details.\n    gam : real\n        Tuning parameter, see references for details.\n    maxiter : int\n        The maximum number of iterations; returns Nones if convergence\n        does not occur by this point\n\n    Returns\n    -------\n    alpha : real\n        The step value\n    x : Array_like\n        The function argument at the final step\n    obval : Real\n        The function value at the final step\n    g : Array_like\n        The gradient at the final step\n\n    Notes\n    -----\n    The basic idea is to take a big step in the direction of the\n    gradient, even if the function value is not decreased (but there\n    is a maximum allowed increase in terms of the recent history of\n    the iterates).\n\n    References\n    ----------\n    Grippo L, Lampariello F, Lucidi S (1986). A Nonmonotone Line\n    Search Technique for Newton's Method. SIAM Journal on Numerical\n    Analysis, 23, 707-716.\n\n    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected\n    gradient methods: Review and perspectives. Journal of Statistical\n    Software (preprint).\n    "
    alpha = 1.0
    last_obval = obj(x)
    obj_max = max(obj_hist[-M:])
    for iter in range(maxiter):
        obval = obj(x + alpha * d)
        g = grad(x)
        gtd = (g * d).sum()
        if obval <= obj_max + gam * alpha * gtd:
            return (alpha, x + alpha * d, obval, g)
        a1 = -0.5 * alpha ** 2 * gtd / (obval - last_obval - alpha * gtd)
        if sig1 <= a1 and a1 <= sig2 * alpha:
            alpha = a1
        else:
            alpha /= 2.0
        last_obval = obval
    return (None, None, None, None)

def _spg_optim(func, grad, start, project, maxiter=10000.0, M=10, ctol=0.001, maxiter_nmls=200, lam_min=1e-30, lam_max=1e+30, sig1=0.1, sig2=0.9, gam=0.0001):
    if False:
        for i in range(10):
            print('nop')
    '\n    Implements the spectral projected gradient method for minimizing a\n    differentiable function on a convex domain.\n\n    Parameters\n    ----------\n    func : real valued function\n        The objective function to be minimized.\n    grad : real array-valued function\n        The gradient of the objective function\n    start : array_like\n        The starting point\n    project : function\n        In-place projection of the argument to the domain\n        of func.\n    ... See notes regarding additional arguments\n\n    Returns\n    -------\n    rslt : Bunch\n        rslt.params is the final iterate, other fields describe\n        convergence status.\n\n    Notes\n    -----\n    This can be an effective heuristic algorithm for problems where no\n    guaranteed algorithm for computing a global minimizer is known.\n\n    There are a number of tuning parameters, but these generally\n    should not be changed except for `maxiter` (positive integer) and\n    `ctol` (small positive real).  See the Birgin et al reference for\n    more information about the tuning parameters.\n\n    Reference\n    ---------\n    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected\n    gradient methods: Review and perspectives. Journal of Statistical\n    Software (preprint).  Available at:\n    http://www.ime.usp.br/~egbirgin/publications/bmr5.pdf\n    '
    lam = min(10 * lam_min, lam_max)
    params = start.copy()
    gval = grad(params)
    obj_hist = [func(params)]
    for itr in range(int(maxiter)):
        df = params - gval
        project(df)
        df -= params
        if np.max(np.abs(df)) < ctol:
            return Bunch(**{'Converged': True, 'params': params, 'objective_values': obj_hist, 'Message': 'Converged successfully'})
        d = params - lam * gval
        project(d)
        d -= params
        (alpha, params1, fval, gval1) = _nmono_linesearch(func, grad, params, d, obj_hist, M=M, sig1=sig1, sig2=sig2, gam=gam, maxiter=maxiter_nmls)
        if alpha is None:
            return Bunch(**{'Converged': False, 'params': params, 'objective_values': obj_hist, 'Message': 'Failed in nmono_linesearch'})
        obj_hist.append(fval)
        s = params1 - params
        y = gval1 - gval
        sy = (s * y).sum()
        if sy <= 0:
            lam = lam_max
        else:
            ss = (s * s).sum()
            lam = max(lam_min, min(ss / sy, lam_max))
        params = params1
        gval = gval1
    return Bunch(**{'Converged': False, 'params': params, 'objective_values': obj_hist, 'Message': 'spg_optim did not converge'})

def _project_correlation_factors(X):
    if False:
        print('Hello World!')
    '\n    Project a matrix into the domain of matrices whose row-wise sums\n    of squares are less than or equal to 1.\n\n    The input matrix is modified in-place.\n    '
    nm = np.sqrt((X * X).sum(1))
    ii = np.flatnonzero(nm > 1)
    if len(ii) > 0:
        X[ii, :] /= nm[ii][:, None]

class FactoredPSDMatrix:
    """
    Representation of a positive semidefinite matrix in factored form.

    The representation is constructed based on a vector `diag` and
    rectangular matrix `root`, such that the PSD matrix represented by
    the class instance is Diag + root * root', where Diag is the
    square diagonal matrix with `diag` on its main diagonal.

    Parameters
    ----------
    diag : 1d array_like
        See above
    root : 2d array_like
        See above

    Notes
    -----
    The matrix is represented internally in the form Diag^{1/2}(I +
    factor * scales * factor')Diag^{1/2}, where `Diag` and `scales`
    are diagonal matrices, and `factor` is an orthogonal matrix.
    """

    def __init__(self, diag, root):
        if False:
            while True:
                i = 10
        self.diag = diag
        self.root = root
        root = root / np.sqrt(diag)[:, None]
        (u, s, vt) = np.linalg.svd(root, 0)
        self.factor = u
        self.scales = s ** 2

    def to_matrix(self):
        if False:
            return 10
        '\n        Returns the PSD matrix represented by this instance as a full\n        (square) matrix.\n        '
        return np.diag(self.diag) + np.dot(self.root, self.root.T)

    def decorrelate(self, rhs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Decorrelate the columns of `rhs`.\n\n        Parameters\n        ----------\n        rhs : array_like\n            A 2 dimensional array with the same number of rows as the\n            PSD matrix represented by the class instance.\n\n        Returns\n        -------\n        C^{-1/2} * rhs, where C is the covariance matrix represented\n        by this class instance.\n\n        Notes\n        -----\n        The returned matrix has the identity matrix as its row-wise\n        population covariance matrix.\n\n        This function exploits the factor structure for efficiency.\n        '
        qval = -1 + 1 / np.sqrt(1 + self.scales)
        rhs = rhs / np.sqrt(self.diag)[:, None]
        rhs1 = np.dot(self.factor.T, rhs)
        rhs1 *= qval[:, None]
        rhs1 = np.dot(self.factor, rhs1)
        rhs += rhs1
        return rhs

    def solve(self, rhs):
        if False:
            while True:
                i = 10
        '\n        Solve a linear system of equations with factor-structured\n        coefficients.\n\n        Parameters\n        ----------\n        rhs : array_like\n            A 2 dimensional array with the same number of rows as the\n            PSD matrix represented by the class instance.\n\n        Returns\n        -------\n        C^{-1} * rhs, where C is the covariance matrix represented\n        by this class instance.\n\n        Notes\n        -----\n        This function exploits the factor structure for efficiency.\n        '
        qval = -self.scales / (1 + self.scales)
        dr = np.sqrt(self.diag)
        rhs = rhs / dr[:, None]
        mat = qval[:, None] * np.dot(self.factor.T, rhs)
        rhs = rhs + np.dot(self.factor, mat)
        return rhs / dr[:, None]

    def logdet(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the logarithm of the determinant of a\n        factor-structured matrix.\n        '
        logdet = np.sum(np.log(self.diag))
        logdet += np.sum(np.log(self.scales))
        logdet += np.sum(np.log(1 + 1 / self.scales))
        return logdet

def corr_nearest_factor(corr, rank, ctol=1e-06, lam_min=1e-30, lam_max=1e+30, maxiter=1000):
    if False:
        print('Hello World!')
    "\n    Find the nearest correlation matrix with factor structure to a\n    given square matrix.\n\n    Parameters\n    ----------\n    corr : square array\n        The target matrix (to which the nearest correlation matrix is\n        sought).  Must be square, but need not be positive\n        semidefinite.\n    rank : int\n        The rank of the factor structure of the solution, i.e., the\n        number of linearly independent columns of X.\n    ctol : positive real\n        Convergence criterion.\n    lam_min : float\n        Tuning parameter for spectral projected gradient optimization\n        (smallest allowed step in the search direction).\n    lam_max : float\n        Tuning parameter for spectral projected gradient optimization\n        (largest allowed step in the search direction).\n    maxiter : int\n        Maximum number of iterations in spectral projected gradient\n        optimization.\n\n    Returns\n    -------\n    rslt : Bunch\n        rslt.corr is a FactoredPSDMatrix defining the estimated\n        correlation structure.  Other fields of `rslt` contain\n        returned values from spg_optim.\n\n    Notes\n    -----\n    A correlation matrix has factor structure if it can be written in\n    the form I + XX' - diag(XX'), where X is n x k with linearly\n    independent columns, and with each row having sum of squares at\n    most equal to 1.  The approximation is made in terms of the\n    Frobenius norm.\n\n    This routine is useful when one has an approximate correlation\n    matrix that is not positive semidefinite, and there is need to\n    estimate the inverse, square root, or inverse square root of the\n    population correlation matrix.  The factor structure allows these\n    tasks to be done without constructing any n x n matrices.\n\n    This is a non-convex problem with no known guaranteed globally\n    convergent algorithm for computing the solution.  Borsdof, Higham\n    and Raydan (2010) compared several methods for this problem and\n    found the spectral projected gradient (SPG) method (used here) to\n    perform best.\n\n    The input matrix `corr` can be a dense numpy array or any scipy\n    sparse matrix.  The latter is useful if the input matrix is\n    obtained by thresholding a very large sample correlation matrix.\n    If `corr` is sparse, the calculations are optimized to save\n    memory, so no working matrix with more than 10^6 elements is\n    constructed.\n\n    References\n    ----------\n    .. [*] R Borsdof, N Higham, M Raydan (2010).  Computing a nearest\n       correlation matrix with factor structure. SIAM J Matrix Anal Appl,\n       31:5, 2603-2622.\n       http://eprints.ma.man.ac.uk/1523/01/covered/MIMS_ep2009_87.pdf\n\n    Examples\n    --------\n    Hard thresholding a correlation matrix may result in a matrix that\n    is not positive semidefinite.  We can approximate a hard\n    thresholded correlation matrix with a PSD matrix as follows, where\n    `corr` is the input correlation matrix.\n\n    >>> import numpy as np\n    >>> from statsmodels.stats.correlation_tools import corr_nearest_factor\n    >>> np.random.seed(1234)\n    >>> b = 1.5 - np.random.rand(10, 1)\n    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)\n    >>> corr = np.corrcoef(x.T)\n    >>> corr = corr * (np.abs(corr) >= 0.3)\n    >>> rslt = corr_nearest_factor(corr, 3)\n    "
    (p, _) = corr.shape
    (u, s, vt) = svds(corr, rank)
    X = u * np.sqrt(s)
    nm = np.sqrt((X ** 2).sum(1))
    ii = np.flatnonzero(nm > 1e-05)
    X[ii, :] /= nm[ii][:, None]
    corr1 = corr.copy()
    if type(corr1) is np.ndarray:
        np.fill_diagonal(corr1, 0)
    elif sparse.issparse(corr1):
        corr1.setdiag(np.zeros(corr1.shape[0]))
        corr1.eliminate_zeros()
        corr1.sort_indices()
    else:
        raise ValueError('Matrix type not supported')

    def grad(X):
        if False:
            i = 10
            return i + 15
        gr = np.dot(X, np.dot(X.T, X))
        if type(corr1) is np.ndarray:
            gr -= np.dot(corr1, X)
        else:
            gr -= corr1.dot(X)
        gr -= (X * X).sum(1)[:, None] * X
        return 4 * gr

    def func(X):
        if False:
            for i in range(10):
                print('nop')
        if type(corr1) is np.ndarray:
            M = np.dot(X, X.T)
            np.fill_diagonal(M, 0)
            M -= corr1
            fval = (M * M).sum()
            return fval
        else:
            fval = 0.0
            max_ws = 1000000.0
            bs = int(max_ws / X.shape[0])
            ir = 0
            while ir < X.shape[0]:
                ir2 = min(ir + bs, X.shape[0])
                u = np.dot(X[ir:ir2, :], X.T)
                ii = np.arange(u.shape[0])
                u[ii, ir + ii] = 0
                u -= np.asarray(corr1[ir:ir2, :].todense())
                fval += (u * u).sum()
                ir += bs
            return fval
    rslt = _spg_optim(func, grad, X, _project_correlation_factors, ctol=ctol, lam_min=lam_min, lam_max=lam_max, maxiter=maxiter)
    root = rslt.params
    diag = 1 - (root ** 2).sum(1)
    soln = FactoredPSDMatrix(diag, root)
    rslt.corr = soln
    del rslt.params
    return rslt

def cov_nearest_factor_homog(cov, rank):
    if False:
        for i in range(10):
            print('nop')
    "\n    Approximate an arbitrary square matrix with a factor-structured\n    matrix of the form k*I + XX'.\n\n    Parameters\n    ----------\n    cov : array_like\n        The input array, must be square but need not be positive\n        semidefinite\n    rank : int\n        The rank of the fitted factor structure\n\n    Returns\n    -------\n    A FactoredPSDMatrix instance containing the fitted matrix\n\n    Notes\n    -----\n    This routine is useful if one has an estimated covariance matrix\n    that is not SPD, and the ultimate goal is to estimate the inverse,\n    square root, or inverse square root of the true covariance\n    matrix. The factor structure allows these tasks to be performed\n    without constructing any n x n matrices.\n\n    The calculations use the fact that if k is known, then X can be\n    determined from the eigen-decomposition of cov - k*I, which can\n    in turn be easily obtained form the eigen-decomposition of `cov`.\n    Thus the problem can be reduced to a 1-dimensional search for k\n    that does not require repeated eigen-decompositions.\n\n    If the input matrix is sparse, then cov - k*I is also sparse, so\n    the eigen-decomposition can be done efficiently using sparse\n    routines.\n\n    The one-dimensional search for the optimal value of k is not\n    convex, so a local minimum could be obtained.\n\n    Examples\n    --------\n    Hard thresholding a covariance matrix may result in a matrix that\n    is not positive semidefinite.  We can approximate a hard\n    thresholded covariance matrix with a PSD matrix as follows:\n\n    >>> import numpy as np\n    >>> np.random.seed(1234)\n    >>> b = 1.5 - np.random.rand(10, 1)\n    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)\n    >>> cov = np.cov(x)\n    >>> cov = cov * (np.abs(cov) >= 0.3)\n    >>> rslt = cov_nearest_factor_homog(cov, 3)\n    "
    (m, n) = cov.shape
    (Q, Lambda, _) = svds(cov, rank)
    if sparse.issparse(cov):
        QSQ = np.dot(Q.T, cov.dot(Q))
        ts = cov.diagonal().sum()
        tss = cov.dot(cov).diagonal().sum()
    else:
        QSQ = np.dot(Q.T, np.dot(cov, Q))
        ts = np.trace(cov)
        tss = np.trace(np.dot(cov, cov))

    def fun(k):
        if False:
            print('Hello World!')
        Lambda_t = Lambda - k
        v = tss + m * k ** 2 + np.sum(Lambda_t ** 2) - 2 * k * ts
        v += 2 * k * np.sum(Lambda_t) - 2 * np.sum(np.diag(QSQ) * Lambda_t)
        return v
    k_opt = fminbound(fun, 0, 100000.0)
    Lambda_opt = Lambda - k_opt
    fac_opt = Q * np.sqrt(Lambda_opt)
    diag = k_opt * np.ones(m, dtype=np.float64)
    return FactoredPSDMatrix(diag, fac_opt)

def corr_thresholded(data, minabs=None, max_elt=10000000.0):
    if False:
        while True:
            i = 10
    '\n    Construct a sparse matrix containing the thresholded row-wise\n    correlation matrix from a data array.\n\n    Parameters\n    ----------\n    data : array_like\n        The data from which the row-wise thresholded correlation\n        matrix is to be computed.\n    minabs : non-negative real\n        The threshold value; correlation coefficients smaller in\n        magnitude than minabs are set to zero.  If None, defaults\n        to 1 / sqrt(n), see Notes for more information.\n\n    Returns\n    -------\n    cormat : sparse.coo_matrix\n        The thresholded correlation matrix, in COO format.\n\n    Notes\n    -----\n    This is an alternative to C = np.corrcoef(data); C \\*= (np.abs(C)\n    >= absmin), suitable for very tall data matrices.\n\n    If the data are jointly Gaussian, the marginal sampling\n    distributions of the elements of the sample correlation matrix are\n    approximately Gaussian with standard deviation 1 / sqrt(n).  The\n    default value of ``minabs`` is thus equal to 1 standard error, which\n    will set to zero approximately 68% of the estimated correlation\n    coefficients for which the population value is zero.\n\n    No intermediate matrix with more than ``max_elt`` values will be\n    constructed.  However memory use could still be high if a large\n    number of correlation values exceed `minabs` in magnitude.\n\n    The thresholded matrix is returned in COO format, which can easily\n    be converted to other sparse formats.\n\n    Examples\n    --------\n    Here X is a tall data matrix (e.g. with 100,000 rows and 50\n    columns).  The row-wise correlation matrix of X is calculated\n    and stored in sparse form, with all entries smaller than 0.3\n    treated as 0.\n\n    >>> import numpy as np\n    >>> np.random.seed(1234)\n    >>> b = 1.5 - np.random.rand(10, 1)\n    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)\n    >>> cmat = corr_thresholded(x, 0.3)\n    '
    (nrow, ncol) = data.shape
    if minabs is None:
        minabs = 1.0 / float(ncol)
    data = data.copy()
    data -= data.mean(1)[:, None]
    sd = data.std(1, ddof=1)
    ii = np.flatnonzero(sd > 1e-05)
    data[ii, :] /= sd[ii][:, None]
    ii = np.flatnonzero(sd <= 1e-05)
    data[ii, :] = 0
    bs = int(np.floor(max_elt / nrow))
    (ipos_all, jpos_all, cor_values) = ([], [], [])
    ir = 0
    while ir < nrow:
        ir2 = min(data.shape[0], ir + bs)
        cm = np.dot(data[ir:ir2, :], data.T) / (ncol - 1)
        cma = np.abs(cm)
        (ipos, jpos) = np.nonzero(cma >= minabs)
        ipos_all.append(ipos + ir)
        jpos_all.append(jpos)
        cor_values.append(cm[ipos, jpos])
        ir += bs
    ipos = np.concatenate(ipos_all)
    jpos = np.concatenate(jpos_all)
    cor_values = np.concatenate(cor_values)
    cmat = sparse.coo_matrix((cor_values, (ipos, jpos)), (nrow, nrow))
    return cmat

class MultivariateKernel:
    """
    Base class for multivariate kernels.

    An instance of MultivariateKernel implements a `call` method having
    signature `call(x, loc)`, returning the kernel weights comparing `x`
    (a 1d ndarray) to each row of `loc` (a 2d ndarray).
    """

    def call(self, x, loc):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def set_bandwidth(self, bw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the bandwidth to the given vector.\n\n        Parameters\n        ----------\n        bw : array_like\n            A vector of non-negative bandwidth values.\n        '
        self.bw = bw
        self._setup()

    def _setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.bwk = np.prod(self.bw)
        self.bw2 = self.bw * self.bw

    def set_default_bw(self, loc, bwm=None):
        if False:
            print('Hello World!')
        '\n        Set default bandwiths based on domain values.\n\n        Parameters\n        ----------\n        loc : array_like\n            Values from the domain to which the kernel will\n            be applied.\n        bwm : scalar, optional\n            A non-negative scalar that is used to multiply\n            the default bandwidth.\n        '
        sd = loc.std(0)
        (q25, q75) = np.percentile(loc, [25, 75], axis=0)
        iqr = (q75 - q25) / 1.349
        bw = np.where(iqr < sd, iqr, sd)
        bw *= 0.9 / loc.shape[0] ** 0.2
        if bwm is not None:
            bw *= bwm
        self.bw = np.asarray(bw, dtype=np.float64)
        self._setup()

class GaussianMultivariateKernel(MultivariateKernel):
    """
    The Gaussian (squared exponential) multivariate kernel.
    """

    def call(self, x, loc):
        if False:
            return 10
        return np.exp(-(x - loc) ** 2 / (2 * self.bw2)).sum(1) / self.bwk

def kernel_covariance(exog, loc, groups, kernel=None, bw=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Use kernel averaging to estimate a multivariate covariance function.\n\n    The goal is to estimate a covariance function C(x, y) =\n    cov(Z(x), Z(y)) where x, y are vectors in R^p (e.g. representing\n    locations in time or space), and Z(.) represents a multivariate\n    process on R^p.\n\n    The data used for estimation can be observed at arbitrary values of the\n    position vector, and there can be multiple independent observations\n    from the process.\n\n    Parameters\n    ----------\n    exog : array_like\n        The rows of exog are realizations of the process obtained at\n        specified points.\n    loc : array_like\n        The rows of loc are the locations (e.g. in space or time) at\n        which the rows of exog are observed.\n    groups : array_like\n        The values of groups are labels for distinct independent copies\n        of the process.\n    kernel : MultivariateKernel instance, optional\n        An instance of MultivariateKernel, defaults to\n        GaussianMultivariateKernel.\n    bw : array_like or scalar\n        A bandwidth vector, or bandwidth multiplier.  If a 1d array, it\n        contains kernel bandwidths for each component of the process, and\n        must have length equal to the number of columns of exog.  If a scalar,\n        bw is a bandwidth multiplier used to adjust the default bandwidth; if\n        None, a default bandwidth is used.\n\n    Returns\n    -------\n    A real-valued function C(x, y) that returns an estimate of the covariance\n    between values of the process located at x and y.\n\n    References\n    ----------\n    .. [1] Genton M, W Kleiber (2015).  Cross covariance functions for\n        multivariate geostatics.  Statistical Science 30(2).\n        https://arxiv.org/pdf/1507.08017.pdf\n    '
    exog = np.asarray(exog)
    loc = np.asarray(loc)
    groups = np.asarray(groups)
    if loc.ndim == 1:
        loc = loc[:, None]
    v = [exog.shape[0], loc.shape[0], len(groups)]
    if min(v) != max(v):
        msg = 'exog, loc, and groups must have the same number of rows'
        raise ValueError(msg)
    ix = {}
    for (i, g) in enumerate(groups):
        if g not in ix:
            ix[g] = []
        ix[g].append(i)
    for g in ix.keys():
        ix[g] = np.sort(ix[g])
    if kernel is None:
        kernel = GaussianMultivariateKernel()
    if bw is None:
        kernel.set_default_bw(loc)
    elif np.isscalar(bw):
        kernel.set_default_bw(loc, bwm=bw)
    else:
        kernel.set_bandwidth(bw)

    def cov(x, y):
        if False:
            return 10
        kx = kernel.call(x, loc)
        ky = kernel.call(y, loc)
        (cm, cw) = (0.0, 0.0)
        for (g, ii) in ix.items():
            m = len(ii)
            (j1, j2) = np.indices((m, m))
            j1 = ii[j1.flat]
            j2 = ii[j2.flat]
            w = kx[j1] * ky[j2]
            cm += np.einsum('ij,ik,i->jk', exog[j1, :], exog[j2, :], w)
            cw += w.sum()
        if cw < 1e-10:
            msg = 'Effective sample size is 0.  The bandwidth may be too ' + 'small, or you are outside the range of your data.'
            warnings.warn(msg)
            return np.nan * np.ones_like(cm)
        return cm / cw
    return cov