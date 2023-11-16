"""
Utility functions for debugging and testing.
"""
from typing import Tuple
import numpy as np
from scipy.stats import unitary_group
import qiskit.transpiler.synthesis.aqc.fast_gradient.fast_grad_utils as fgu

def relative_error(a_mat: np.ndarray, b_mat: np.ndarray) -> float:
    if False:
        print('Hello World!')
    '\n    Computes relative residual between two matrices in Frobenius norm.\n    '
    return float(np.linalg.norm(a_mat - b_mat, 'fro')) / float(np.linalg.norm(b_mat, 'fro'))

def make_unit_vector(pos: int, nbits: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Makes a unit vector ``e = (0 ... 0 1 0 ... 0)`` of size ``2^n`` with\n    unit at the position ``num``. **Note**: result depends on bit ordering.\n\n    Args:\n        pos: position of unit in vector.\n        nbits: number of meaningful bit in the number "pos".\n\n    Returns:\n        unit vector of size ``2^n``.\n    '
    vec = np.zeros((2 ** nbits,), dtype=np.int64)
    vec[fgu.reverse_bits(pos, nbits, enable=True)] = 1
    return vec

def eye_int(n: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Creates an identity matrix with integer entries.\n\n    Args:\n        n: number of bits.\n\n    Returns:\n        unit matrix of size ``2^n`` with integer entries.\n    '
    return np.eye(2 ** n, dtype=np.int64)

def kron3(a_mat: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Computes Kronecker product of 3 matrices.\n    '
    return np.kron(a_mat, np.kron(b_mat, c_mat))

def kron5(a_mat: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray, d_mat: np.ndarray, e_mat: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Computes Kronecker product of 5 matrices.\n    '
    return np.kron(np.kron(np.kron(np.kron(a_mat, b_mat), c_mat), d_mat), e_mat)

def rand_matrix(dim: int, kind: str='complex') -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Generates a random complex or integer value matrix.\n\n    Args:\n        dim: matrix size dim-x-dim.\n        kind: "complex" or "randint".\n\n    Returns:\n        a random matrix.\n    '
    if kind == 'complex':
        return np.random.rand(dim, dim).astype(np.complex128) + np.random.rand(dim, dim).astype(np.complex128) * 1j
    else:
        return np.random.randint(low=1, high=100, size=(dim, dim), dtype=np.int64)

def make_test_matrices2x2(n: int, k: int, kind: str='complex') -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Creates a ``2^n x 2^n`` random matrix made as a Kronecker product of identity\n    ones and a single 1-qubit gate. This models a layer in quantum circuit with\n    an arbitrary 1-qubit gate somewhere in the middle.\n\n    Args:\n        n: number of qubits.\n        k: index of qubit a 1-qubit gate is acting on.\n        kind: entries of the output matrix are defined as:\n              "complex", "primes" or "randint".\n    Returns:\n        A tuple of (1) ``2^n x 2^n`` random matrix; (2) ``2 x 2`` matrix of 1-qubit\n          gate used for matrix construction.\n    '
    if kind == 'primes':
        a_mat = np.asarray([[2, 3], [5, 7]], dtype=np.int64)
    else:
        a_mat = rand_matrix(dim=2, kind=kind)
    m_mat = kron3(eye_int(k), a_mat, eye_int(n - k - 1))
    return (m_mat, a_mat)

def make_test_matrices4x4(n: int, j: int, k: int, kind: str='complex') -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    Creates a ``2^n x 2^n`` random matrix made as a Kronecker product of identity\n    ones and a single 2-qubit gate. This models a layer in quantum circuit with\n    an arbitrary 2-qubit gate somewhere in the middle.\n\n    Args:\n        n: number of qubits.\n        j: index of the first qubit the 2-qubit gate acting on.\n        k: index of the second qubit the 2-qubit gate acting on.\n        kind: entries of the output matrix are defined as:\n              "complex", "primes" or "randint".\n\n    Returns:\n        A tuple of (1) ``2^n x 2^n`` random matrix; (2) ``4 x 4`` matrix of\n        2-qubit gate used for matrix construction.\n    '
    if kind == 'primes':
        a_mat = np.asarray([[2, 3], [5, 7]], dtype=np.int64)
        b_mat = np.asarray([[11, 13], [17, 19]], dtype=np.int64)
        c_mat = np.asarray([[47, 53], [41, 43]], dtype=np.int64)
        d_mat = np.asarray([[31, 37], [23, 29]], dtype=np.int64)
    else:
        (a_mat, b_mat) = (rand_matrix(dim=2, kind=kind), rand_matrix(dim=2, kind=kind))
        (c_mat, d_mat) = (rand_matrix(dim=2, kind=kind), rand_matrix(dim=2, kind=kind))
    if j < k:
        m_mat = kron5(eye_int(j), a_mat, eye_int(k - j - 1), b_mat, eye_int(n - k - 1)) + kron5(eye_int(j), c_mat, eye_int(k - j - 1), d_mat, eye_int(n - k - 1))
    else:
        m_mat = kron5(eye_int(k), b_mat, eye_int(j - k - 1), a_mat, eye_int(n - j - 1)) + kron5(eye_int(k), d_mat, eye_int(j - k - 1), c_mat, eye_int(n - j - 1))
    g_mat = np.kron(a_mat, b_mat) + np.kron(c_mat, d_mat)
    return (m_mat, g_mat)

def rand_circuit(num_qubits: int, depth: int) -> np.ndarray:
    if False:
        return 10
    '\n    Generates a random circuit of unit blocks for debugging and testing.\n    '
    blocks = np.tile(np.arange(num_qubits).reshape(num_qubits, 1), depth)
    for i in range(depth):
        np.random.shuffle(blocks[:, i])
    return blocks[0:2, :].copy()

def rand_su_mat(dim: int) -> np.ndarray:
    if False:
        return 10
    '\n    Generates a random SU matrix.\n    Args:\n        dim: matrix size ``dim-x-dim``.\n    Returns:\n        random SU matrix.\n    '
    u_mat = unitary_group.rvs(dim)
    u_mat /= np.linalg.det(u_mat) ** (1.0 / float(dim))
    return u_mat