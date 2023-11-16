"""Routines that use local invariants to compute properties
of two-qubit unitary operators.
"""
from __future__ import annotations
from math import sqrt
import numpy as np
INVARIANT_TOL = 1e-12
MAGIC = 1.0 / sqrt(2) * np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex)

def two_qubit_local_invariants(U: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    'Computes the local invariants for a two-qubit unitary.\n\n    Args:\n        U (ndarray): Input two-qubit unitary.\n\n    Returns:\n        ndarray: NumPy array of local invariants [g0, g1, g2].\n\n    Raises:\n        ValueError: Input not a 2q unitary.\n\n    Notes:\n        Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).\n        Zhang et al., Phys Rev A. 67, 042313 (2003).\n    '
    U = np.asarray(U)
    if U.shape != (4, 4):
        raise ValueError('Unitary must correspond to a two-qubit gate.')
    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    det_um = np.linalg.det(Um)
    M = Um.T.dot(Um)
    m_tr2 = M.trace()
    m_tr2 *= m_tr2
    G1 = m_tr2 / (16 * det_um)
    G2 = (m_tr2 - np.trace(M.dot(M))) / (4 * det_um)
    return np.round([G1.real, G1.imag, G2.real], 12) + 0.0

def local_equivalence(weyl: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Computes the equivalent local invariants from the\n    Weyl coordinates.\n\n    Args:\n        weyl (ndarray): Weyl coordinates.\n\n    Returns:\n        ndarray: Local equivalent coordinates [g0, g1, g3].\n\n    Notes:\n        This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003),\n        but we multiply weyl coordinates by 2 since we are\n        working in the reduced chamber.\n    '
    g0_equiv = np.prod(np.cos(2 * weyl) ** 2) - np.prod(np.sin(2 * weyl) ** 2)
    g1_equiv = np.prod(np.sin(4 * weyl)) / 4
    g2_equiv = 4 * np.prod(np.cos(2 * weyl) ** 2) - 4 * np.prod(np.sin(2 * weyl) ** 2) - np.prod(np.cos(4 * weyl))
    return np.round([g0_equiv, g1_equiv, g2_equiv], 12) + 0.0