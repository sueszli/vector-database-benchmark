"""
Utility functions in the fast gradient implementation.
"""
from __future__ import annotations
from typing import Union
import numpy as np

def is_permutation(x: np.ndarray) -> bool:
    if False:
        return 10
    '\n    Checks if array is really an index permutation.\n\n    Args:\n        1D-array of integers that supposedly represents a permutation.\n\n    Returns:\n        True, if array is really a permutation of indices.\n    '
    return isinstance(x, np.ndarray) and x.ndim == 1 and (x.dtype == np.int64) and np.all(np.sort(x) == np.arange(x.size, dtype=np.int64))

def reverse_bits(x: Union[int, np.ndarray], nbits: int, enable: bool) -> Union[int, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Reverses the bit order in a number of ``nbits`` length.\n    If ``x`` is an array, then operation is applied to every entry.\n\n    Args:\n        x: either a single integer or an array of integers.\n        nbits: number of meaningful bits in the number x.\n        enable: apply reverse operation, if enabled, otherwise leave unchanged.\n\n    Returns:\n        a number or array of numbers with reversed bits.\n    '
    if not enable:
        if isinstance(x, int):
            pass
        else:
            x = x.copy()
        return x
    if isinstance(x, int):
        res: int | np.ndarray = int(0)
    else:
        x = x.copy()
        res = np.full_like(x, fill_value=0)
    for _ in range(nbits):
        res <<= 1
        res |= x & 1
        x >>= 1
    return res

def swap_bits(num: int, a: int, b: int) -> int:
    if False:
        while True:
            i = 10
    "\n    Swaps the bits at positions 'a' and 'b' in the number 'num'.\n\n    Args:\n        num: an integer number where bits should be swapped.\n        a: index of the first bit to be swapped.\n        b: index of the second bit to be swapped.\n\n    Returns:\n        the number with swapped bits.\n    "
    x = (num >> a ^ num >> b) & 1
    return num ^ (x << a | x << b)

def bit_permutation_1q(n: int, k: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Constructs index permutation that brings a circuit consisting of a single\n    1-qubit gate to "standard form": ``kron(I(2^n/2), G)``, as we call it. Here n\n    is the number of qubits, ``G`` is a 2x2 gate matrix, ``I(2^n/2)`` is the identity\n    matrix of size ``(2^n/2)x(2^n/2)``, and the full size of the circuit matrix is\n    ``(2^n)x(2^n)``. Circuit matrix in standard form becomes block-diagonal (with\n    sub-matrices ``G`` on the main diagonal). Multiplication of such a matrix and\n    a dense one is much faster than generic dense-dense product. Moreover,\n    we do not need to keep the entire circuit matrix in memory but just 2x2 ``G``\n    one. This saves a lot of memory when the number of qubits is large.\n\n    Args:\n        n: number of qubits.\n        k: index of qubit where single 1-qubit gate is applied.\n\n    Returns:\n        permutation that brings the whole layer to the standard form.\n    '
    perm = np.arange(2 ** n, dtype=np.int64)
    if k != n - 1:
        for v in range(2 ** n):
            perm[v] = swap_bits(v, k, n - 1)
    return perm

def bit_permutation_2q(n: int, j: int, k: int) -> np.ndarray:
    if False:
        return 10
    '\n    Constructs index permutation that brings a circuit consisting of a single\n    2-qubit gate to "standard form": ``kron(I(2^n/4), G)``, as we call it. Here ``n``\n    is the number of qubits, ``G`` is a 4x4 gate matrix, ``I(2^n/4)`` is the identity\n    matrix of size ``(2^n/4)x(2^n/4)``, and the full size of the circuit matrix is\n    ``(2^n)x(2^n)``. Circuit matrix in standard form becomes block-diagonal (with\n    sub-matrices ``G`` on the main diagonal). Multiplication of such a matrix and\n    a dense one is much faster than generic dense-dense product. Moreover,\n    we do not need to keep the entire circuit matrix in memory but just 4x4 ``G``\n    one. This saves a lot of memory when the number of qubits is large.\n\n    Args:\n        n: number of qubits.\n        j: index of control qubit where single 2-qubit gate is applied.\n        k: index of target qubit where single 2-qubit gate is applied.\n\n    Returns:\n        permutation that brings the whole layer to the standard form.\n    '
    dim = 2 ** n
    perm = np.arange(dim, dtype=np.int64)
    if j < n - 2:
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, j, n - 2), k, n - 1)
        elif k == n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, n - 2, n - 1), j, n - 2)
        else:
            for v in range(dim):
                perm[v] = swap_bits(v, j, n - 2)
    elif j == n - 2:
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(v, k, n - 1)
        else:
            pass
    elif k < n - 2:
        for v in range(dim):
            perm[v] = swap_bits(swap_bits(v, n - 2, n - 1), k, n - 1)
    else:
        for v in range(dim):
            perm[v] = swap_bits(v, n - 2, n - 1)
    return perm

def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns inverse permutation.\n\n    Args:\n        perm: permutation to be reversed.\n\n    Returns:\n        inverse permutation.\n    '
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(perm.size, dtype=np.int64)
    return inv

def make_rx(phi: float, out: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Makes a 2x2 matrix that corresponds to X-rotation gate.\n    This is a fast implementation that does not allocate the output matrix.\n\n    Args:\n        phi: rotation angle.\n        out: placeholder for the result (2x2, complex-valued matrix).\n\n    Returns:\n        rotation gate, same object as referenced by "out".\n    '
    a = 0.5 * phi
    (cs, sn) = (np.cos(a).item(), -1j * np.sin(a).item())
    out[0, 0] = cs
    out[0, 1] = sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out

def make_ry(phi: float, out: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Makes a 2x2 matrix that corresponds to Y-rotation gate.\n    This is a fast implementation that does not allocate the output matrix.\n\n    Args:\n        phi: rotation angle.\n        out: placeholder for the result (2x2, complex-valued matrix).\n\n    Returns:\n        rotation gate, same object as referenced by "out".\n    '
    a = 0.5 * phi
    (cs, sn) = (np.cos(a).item(), np.sin(a).item())
    out[0, 0] = cs
    out[0, 1] = -sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out

def make_rz(phi: float, out: np.ndarray) -> np.ndarray:
    if False:
        return 10
    '\n    Makes a 2x2 matrix that corresponds to Z-rotation gate.\n    This is a fast implementation that does not allocate the output matrix.\n\n    Args:\n        phi: rotation angle.\n        out: placeholder for the result (2x2, complex-valued matrix).\n\n    Returns:\n        rotation gate, same object as referenced by "out".\n    '
    exp = np.exp(0.5j * phi).item()
    out[0, 0] = 1.0 / exp
    out[0, 1] = 0
    out[1, 0] = 0
    out[1, 1] = exp
    return out