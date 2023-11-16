import cvxpy.utilities.cpp.sparsecholesky as spchol
import cvxpy.settings as settings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_matrix

def orth(V, tol=1e-12):
    if False:
        while True:
            i = 10
    'Return a matrix whose columns are an orthonormal basis for range(V)'
    (Q, R, p) = la.qr(V, mode='economic', pivoting=True)
    rank = np.count_nonzero(np.sum(np.abs(R) > tol, axis=1))
    Q = Q[:, :rank].reshape((V.shape[0], rank))
    return Q

def onb_for_orthogonal_complement(V):
    if False:
        while True:
            i = 10
    '\n    Let U = the orthogonal complement of range(V).\n\n    This function returns an array Q whose columns are\n    an orthonormal basis for U. It requires that dim(U) > 0.\n    '
    n = V.shape[0]
    Q1 = orth(V)
    rank = Q1.shape[1]
    assert n > rank
    if np.iscomplexobj(V):
        P = np.eye(n) - Q1 @ Q1.conj().T
    else:
        P = np.eye(n) - Q1 @ Q1.T
    Q2 = orth(P)
    return Q2

def is_diagonal(A):
    if False:
        while True:
            i = 10
    if isinstance(A, spar.spmatrix):
        off_diagonal_elements = A - spar.diags(A.diagonal())
        off_diagonal_elements = off_diagonal_elements.toarray()
    elif isinstance(A, np.ndarray):
        off_diagonal_elements = A - np.diag(np.diag(A))
    else:
        raise ValueError('Unsupported matrix type.')
    return np.allclose(off_diagonal_elements, 0)

def is_psd_within_tol(A, tol):
    if False:
        i = 10
        return i + 15
    '\n    Return True if we can certify that A is PSD (up to tolerance "tol").\n\n    First we check if A is PSD according to the Gershgorin Circle Theorem.\n\n    If Gershgorin is inconclusive, then we use an iterative method (from ARPACK,\n    as called through SciPy) to estimate extremal eigenvalues of certain shifted\n    versions of A. The shifts are chosen so that the signs of those eigenvalues\n    tell us the signs of the eigenvalues of A.\n\n    If there are numerical issues then it\'s possible that this function returns\n    False even when A is PSD. If you know that you\'re in that situation, then\n    you should replace A by\n\n        A = cvxpy.atoms.affine.wraps.psd_wrap(A).\n\n    Parameters\n    ----------\n    A : Union[np.ndarray, spar.spmatrix]\n        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse matrix.\n\n    tol : float\n        Nonnegative. Something very small, like 1e-10.\n    '
    if gershgorin_psd_check(A, tol):
        return True
    if is_diagonal(A):
        if isinstance(A, csc_matrix):
            return np.all(A.data >= -tol)
        else:
            min_diag_entry = np.min(np.diag(A))
            return min_diag_entry >= -tol

    def SA_eigsh(sigma):
        if False:
            return 10
        if hasattr(np.random, 'default_rng'):
            g = np.random.default_rng(123)
        else:
            g = np.random.RandomState(123)
        n = A.shape[0]
        v0 = g.normal(loc=0.0, scale=1.0, size=n)
        return sparla.eigsh(A, k=1, sigma=sigma, which='SA', v0=v0, return_eigenvectors=False)
    try:
        ev = SA_eigsh(-tol)
    except sparla.ArpackNoConvergence as e:
        message = "\n        CVXPY note: This failure was encountered while trying to certify\n        that a matrix is positive semi-definite (see [1] for a definition).\n        In rare cases, this method fails for numerical reasons even when the matrix is\n        positive semi-definite. If you know that you're in that situation, you can\n        replace the matrix A by cvxpy.psd_wrap(A).\n\n        [1] https://en.wikipedia.org/wiki/Definite_matrix\n        "
        error_with_note = f'{str(e)}\n\n{message}'
        raise sparla.ArpackNoConvergence(error_with_note, e.eigenvalues, e.eigenvectors)
    if np.isnan(ev).any():
        temp = tol - np.finfo(A.dtype).eps
        ev = SA_eigsh(-temp)
    return np.all(ev >= -tol)

def gershgorin_psd_check(A, tol):
    if False:
        return 10
    '\n    Use the Gershgorin Circle Theorem\n\n        https://en.wikipedia.org/wiki/Gershgorin_circle_theorem\n\n    As a sufficient condition for A being PSD with tolerance "tol".\n\n    The computational complexity of this function is O(nnz(A)).\n\n    Parameters\n    ----------\n    A : Union[np.ndarray, spar.spmatrix]\n        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse matrix.\n\n    tol : float\n        Nonnegative. Something very small, like 1e-10.\n\n    Returns\n    -------\n    True if A is PSD according to the Gershgorin Circle Theorem.\n    Otherwise, return False.\n    '
    if isinstance(A, spar.spmatrix):
        diag = A.diagonal()
        if np.any(diag < -tol):
            return False
        A_shift = A - spar.diags(diag)
        A_shift = np.abs(A_shift)
        radii = np.array(A_shift.sum(axis=0)).ravel()
        return np.all(diag - radii >= -tol)
    elif isinstance(A, np.ndarray):
        diag = np.diag(A)
        if np.any(diag < -tol):
            return False
        A_shift = A - np.diag(diag)
        A_shift = np.abs(A_shift)
        radii = A_shift.sum(axis=0)
        return np.all(diag - radii >= -tol)
    else:
        raise ValueError()

class SparseCholeskyMessages:
    ASYMMETRIC = 'Input matrix is not symmetric to within provided tolerance.'
    INDEFINITE = 'Input matrix is neither positive nor negative definite.'
    EIGEN_FAIL = 'Cholesky decomposition failed.'
    NOT_SPARSE = 'Input must be a SciPy sparse matrix.'
    NOT_REAL = 'Input matrix must be real.'

def sparse_cholesky(A, sym_tol=settings.CHOL_SYM_TOL, assume_posdef=False):
    if False:
        while True:
            i = 10
    "\n    The input matrix A must be real and symmetric. If A is positive definite then\n    Eigen will be used to compute its sparse Cholesky decomposition with AMD-ordering.\n    If A is negative definite, then the analogous operation will be applied to -A.\n\n    If Cholesky succeeds, then we return a lower-triangular matrix L in\n    CSR-format and a permutation vector p so (L[p, :]) @ (L[p, :]).T == A\n    within numerical precision.\n\n    We raise a ValueError if Eigen's Cholesky fails or if we certify indefiniteness\n    before calling Eigen. While checking for indefiniteness, we also check that\n     ||A - A'||_Fro / sqrt(n) <= sym_tol, where n is the order of the matrix.\n    "
    if not isinstance(A, spar.spmatrix):
        raise ValueError(SparseCholeskyMessages.NOT_SPARSE)
    if np.iscomplexobj(A):
        raise ValueError(SparseCholeskyMessages.NOT_REAL)
    if not assume_posdef:
        symdiff = A - A.T
        sz = symdiff.data.size
        if sz > 0 and la.norm(symdiff.data) > sym_tol * sz ** 0.5:
            raise ValueError(SparseCholeskyMessages.ASYMMETRIC)
        d = A.diagonal()
        maybe_posdef = np.all(d > 0)
        maybe_negdef = np.all(d < 0)
        if not (maybe_posdef or maybe_negdef):
            raise ValueError(SparseCholeskyMessages.INDEFINITE)
        if maybe_negdef:
            (_, L, p) = sparse_cholesky(-A, sym_tol, assume_posdef=True)
            return (-1.0, L, p)
    A_coo = spar.coo_matrix(A)
    n = A.shape[0]
    inrows = spchol.IntVector(A_coo.row)
    incols = spchol.IntVector(A_coo.col)
    invals = spchol.DoubleVector(A_coo.data)
    outpivs = spchol.IntVector()
    outrows = spchol.IntVector()
    outcols = spchol.IntVector()
    outvals = spchol.DoubleVector()
    try:
        spchol.sparse_chol_from_vecs(n, inrows, incols, invals, outpivs, outrows, outcols, outvals)
    except RuntimeError as e:
        if e.args[0] == SparseCholeskyMessages.EIGEN_FAIL:
            raise ValueError(e.args)
        else:
            raise RuntimeError(e.args)
    outvals = np.array(outvals)
    outrows = np.array(outrows)
    outcols = np.array(outcols)
    outpivs = np.array(outpivs)
    L = spar.csr_matrix((outvals, (outrows, outcols)), shape=(n, n))
    return (1.0, L, outpivs)