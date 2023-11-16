"""Utility functions for handling binary matrices."""
from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError

def check_invertible_binary_matrix(mat: np.ndarray):
    if False:
        i = 10
        return i + 15
    'Check that a binary matrix is invertible.\n\n    Args:\n        mat: a binary matrix.\n\n    Returns:\n        bool: True if mat in invertible and False otherwise.\n    '
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        return False
    rank = _compute_rank(mat)
    return rank == mat.shape[0]

def random_invertible_binary_matrix(num_qubits: int, seed: Optional[Union[np.random.Generator, int]]=None):
    if False:
        for i in range(10):
            print('nop')
    'Generates a random invertible n x n binary matrix.\n\n    Args:\n        num_qubits: the matrix size.\n        seed: a random seed.\n\n    Returns:\n        np.ndarray: A random invertible binary matrix of size num_qubits.\n    '
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    rank = 0
    while rank != num_qubits:
        mat = rng.integers(2, size=(num_qubits, num_qubits))
        rank = _compute_rank(mat)
    return mat

def _gauss_elimination(mat, ncols=None, full_elim=False):
    if False:
        print('Hello World!')
    'Gauss elimination of a matrix mat with m rows and n columns.\n    If full_elim = True, it allows full elimination of mat[:, 0 : ncols]\n    Returns the matrix mat.'
    (mat, _) = _gauss_elimination_with_perm(mat, ncols, full_elim)
    return mat

def _gauss_elimination_with_perm(mat, ncols=None, full_elim=False):
    if False:
        i = 10
        return i + 15
    'Gauss elimination of a matrix mat with m rows and n columns.\n    If full_elim = True, it allows full elimination of mat[:, 0 : ncols]\n    Returns the matrix mat, and the permutation perm that was done on the rows during the process.\n    perm[0 : rank] represents the indices of linearly independent rows in the original matrix.'
    mat = np.array(mat, dtype=int, copy=True)
    m = mat.shape[0]
    n = mat.shape[1]
    if ncols is not None:
        n = min(n, ncols)
    perm = np.array(range(m))
    r = 0
    k = 0
    while r < m and k < n:
        is_non_zero = False
        new_r = r
        for j in range(k, n):
            for i in range(r, m):
                if mat[i][j]:
                    is_non_zero = True
                    k = j
                    new_r = i
                    break
            if is_non_zero:
                break
        if not is_non_zero:
            return (mat, perm)
        if new_r != r:
            mat[[r, new_r]] = mat[[new_r, r]]
            (perm[r], perm[new_r]) = (perm[new_r], perm[r])
        if full_elim:
            for i in range(0, r):
                if mat[i][k]:
                    mat[i] = mat[i] ^ mat[r]
        for i in range(r + 1, m):
            if mat[i][k]:
                mat[i] = mat[i] ^ mat[r]
        r += 1
    return (mat, perm)

def calc_inverse_matrix(mat: np.ndarray, verify: bool=False):
    if False:
        return 10
    'Given a square numpy(dtype=int) matrix mat, tries to compute its inverse.\n\n    Args:\n        mat: a boolean square matrix.\n        verify: if True asserts that the multiplication of mat and its inverse is the identity matrix.\n\n    Returns:\n        np.ndarray: the inverse matrix.\n\n    Raises:\n         QiskitError: if the matrix is not square.\n         QiskitError: if the matrix is not invertible.\n    '
    if mat.shape[0] != mat.shape[1]:
        raise QiskitError('Matrix to invert is a non-square matrix.')
    n = mat.shape[0]
    mat1 = np.concatenate((mat, np.eye(n, dtype=int)), axis=1)
    mat1 = _gauss_elimination(mat1, None, full_elim=True)
    r = _compute_rank_after_gauss_elim(mat1[:, 0:n])
    if r < n:
        raise QiskitError('The matrix is not invertible.')
    matinv = mat1[:, n:2 * n]
    if verify:
        mat2 = np.dot(mat, matinv) % 2
        assert np.array_equal(mat2, np.eye(n))
    return matinv

def _compute_rank_after_gauss_elim(mat):
    if False:
        while True:
            i = 10
    'Given a matrix A after Gaussian elimination, computes its rank\n    (i.e. simply the number of nonzero rows)'
    return np.sum(mat.any(axis=1))

def _compute_rank(mat):
    if False:
        for i in range(10):
            print('nop')
    'Given a matrix A computes its rank'
    mat = _gauss_elimination(mat)
    return np.sum(mat.any(axis=1))

def _row_op(mat, ctrl, trgt):
    if False:
        i = 10
        return i + 15
    mat[trgt] = mat[trgt] ^ mat[ctrl]

def _col_op(mat, ctrl, trgt):
    if False:
        return 10
    mat[:, ctrl] = mat[:, trgt] ^ mat[:, ctrl]