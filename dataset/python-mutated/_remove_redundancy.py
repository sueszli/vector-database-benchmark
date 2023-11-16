"""
Routines for removing redundant (linearly dependent) equations from linear
programming equality constraints.
"""
import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm

def _row_count(A):
    if False:
        i = 10
        return i + 15
    '\n    Counts the number of nonzeros in each row of input array A.\n    Nonzeros are defined as any element with absolute value greater than\n    tol = 1e-13. This value should probably be an input to the function.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing a matrix\n\n    Returns\n    -------\n    rowcount : 1-D array\n        Number of nonzeros in each row of A\n\n    '
    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1)).flatten()

def _get_densest(A, eligibleRows):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the index of the densest row of A. Ignores rows that are not\n    eligible for consideration.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing a matrix\n    eligibleRows : 1-D logical array\n        Values indicate whether the corresponding row of A is eligible\n        to be considered\n\n    Returns\n    -------\n    i_densest : int\n        Index of the densest row in A eligible for consideration\n\n    '
    rowCounts = _row_count(A)
    return np.argmax(rowCounts * eligibleRows)

def _remove_zero_rows(A, b):
    if False:
        i = 10
        return i + 15
    '\n    Eliminates trivial equations from system of equations defined by Ax = b\n   and identifies trivial infeasibilities\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the removal operation\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    '
    status = 0
    message = ''
    i_zero = _row_count(A) == 0
    A = A[np.logical_not(i_zero), :]
    if not np.allclose(b[i_zero], 0):
        status = 2
        message = 'There is a zero row in A_eq with a nonzero corresponding entry in b_eq. The problem is infeasible.'
    b = b[np.logical_not(i_zero)]
    return (A, b, status, message)

def bg_update_dense(plu, perm_r, v, j):
    if False:
        return 10
    (LU, p) = plu
    vperm = v[perm_r]
    u = dtrsm(1, LU, vperm, lower=1, diag=1)
    LU[:j + 1, j] = u[:j + 1]
    l = u[j + 1:]
    piv = LU[j, j]
    LU[j + 1:, j] += l / piv
    return (LU, p)

def _remove_redundancy_pivot_dense(A, rhs, true_rank=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D sparse matrix\n        An matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D sparse matrix\n        A matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n\n    '
    tolapiv = 1e-08
    tolprimal = 1e-08
    status = 0
    message = ''
    inconsistent = 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.'
    (A, rhs, status, message) = _remove_zero_rows(A, rhs)
    if status != 0:
        return (A, rhs, status, message)
    (m, n) = A.shape
    v = list(range(m))
    b = list(v)
    d = []
    perm_r = None
    A_orig = A
    A = np.zeros((m, m + n), order='F')
    np.fill_diagonal(A, 1)
    A[:, m:] = A_orig
    e = np.zeros(m)
    js_candidates = np.arange(m, m + n, dtype=int)
    js_mask = np.ones(js_candidates.shape, dtype=bool)
    lu = (np.eye(m, order='F'), np.arange(m))
    perm_r = lu[1]
    for i in v:
        e[i] = 1
        if i > 0:
            e[i - 1] = 0
        try:
            j = b[i - 1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i - 1)
        except Exception:
            lu = scipy.linalg.lu_factor(A[:, b])
            (LU, p) = lu
            perm_r = list(range(m))
            for (i1, i2) in enumerate(p):
                (perm_r[i1], perm_r[i2]) = (perm_r[i2], perm_r[i1])
        pi = scipy.linalg.lu_solve(lu, e, trans=1)
        js = js_candidates[js_mask]
        batch = 50
        for j_index in range(0, len(js), batch):
            j_indices = js[j_index:min(j_index + batch, len(js))]
            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                j = js[j_index + np.argmax(c)]
                b[i] = j
                js_mask[j - m] = False
                break
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar) / (1 + bnorm) > tolprimal:
                status = 2
                message = inconsistent
                return (A_orig, rhs, status, message)
            else:
                d.append(i)
                if true_rank is not None and len(d) == m - true_rank:
                    break
    keep = set(range(m))
    keep = list(keep - set(d))
    return (A_orig[keep, :], rhs[keep], status, message)

def _remove_redundancy_pivot_sparse(A, rhs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D sparse matrix\n        An matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D sparse matrix\n        A matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n\n    '
    tolapiv = 1e-08
    tolprimal = 1e-08
    status = 0
    message = ''
    inconsistent = 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.'
    (A, rhs, status, message) = _remove_zero_rows(A, rhs)
    if status != 0:
        return (A, rhs, status, message)
    (m, n) = A.shape
    v = list(range(m))
    b = list(v)
    k = set(range(m, m + n))
    d = []
    A_orig = A
    A = scipy.sparse.hstack((scipy.sparse.eye(m), A)).tocsc()
    e = np.zeros(m)
    for i in v:
        B = A[:, b]
        e[i] = 1
        if i > 0:
            e[i - 1] = 0
        pi = scipy.sparse.linalg.spsolve(B.transpose(), e).reshape(-1, 1)
        js = list(k - set(b))
        c = (np.abs(A[:, js].transpose().dot(pi)) > tolapiv).nonzero()[0]
        if len(c) > 0:
            j = js[c[0]]
            b[i] = j
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar) / (1 + bnorm) > tolprimal:
                status = 2
                message = inconsistent
                return (A_orig, rhs, status, message)
            else:
                d.append(i)
    keep = set(range(m))
    keep = list(keep - set(d))
    return (A_orig[keep, :], rhs[keep], status, message)

def _remove_redundancy_svd(A, b):
    if False:
        i = 10
        return i + 15
    '\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n\n    '
    (A, b, status, message) = _remove_zero_rows(A, b)
    if status != 0:
        return (A, b, status, message)
    (U, s, Vh) = svd(A)
    eps = np.finfo(float).eps
    tol = s.max() * max(A.shape) * eps
    (m, n) = A.shape
    s_min = s[-1] if m <= n else 0
    while abs(s_min) < tol:
        v = U[:, -1]
        eligibleRows = np.abs(v) > tol * 10000000.0
        if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
            status = 4
            message = 'Due to numerical issues, redundant equality constraints could not be removed automatically. Try providing your constraint matrices as sparse matrices to activate sparse presolve, try turning off redundancy removal, or try turning off presolve altogether.'
            break
        if np.any(np.abs(v.dot(b)) > tol * 100):
            status = 2
            message = 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.'
            break
        i_remove = _get_densest(A, eligibleRows)
        A = np.delete(A, i_remove, axis=0)
        b = np.delete(b, i_remove)
        (U, s, Vh) = svd(A)
        (m, n) = A.shape
        s_min = s[-1] if m <= n else 0
    return (A, b, status, message)

def _remove_redundancy_id(A, rhs, rank=None, randomized=True):
    if False:
        for i in range(10):
            print('nop')
    'Eliminates redundant equations from a system of equations.\n\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n    rank : int, optional\n        The rank of A\n    randomized: bool, optional\n        True for randomized interpolative decomposition\n\n    Returns\n    -------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    '
    status = 0
    message = ''
    inconsistent = 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.'
    (A, rhs, status, message) = _remove_zero_rows(A, rhs)
    if status != 0:
        return (A, rhs, status, message)
    (m, n) = A.shape
    k = rank
    if rank is None:
        k = np.linalg.matrix_rank(A)
    (idx, proj) = interp_decomp(A.T, k, rand=randomized)
    if not np.allclose(rhs[idx[:k]] @ proj, rhs[idx[k:]]):
        status = 2
        message = inconsistent
    idx = sorted(idx[:k])
    A2 = A[idx, :]
    rhs2 = rhs[idx]
    return (A2, rhs2, status, message)