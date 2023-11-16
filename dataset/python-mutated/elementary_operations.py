"""
These are a number of elementary functions that are required for the AQC routines to work.
"""
import numpy as np
from qiskit.circuit.library import RXGate, RZGate, RYGate

def place_unitary(unitary: np.ndarray, n: int, j: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Computes I(j - 1) tensor product U tensor product I(n - j), where U is a unitary matrix\n    of size ``(2, 2)``.\n\n    Args:\n        unitary: a unitary matrix of size ``(2, 2)``.\n        n: num qubits.\n        j: position where to place a unitary.\n\n    Returns:\n        a unitary of n qubits with u in position j.\n    '
    return np.kron(np.kron(np.eye(2 ** j), unitary), np.eye(2 ** (n - 1 - j)))

def place_cnot(n: int, j: int, k: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Places a CNOT from j to k.\n\n    Args:\n        n: number of qubits.\n        j: control qubit.\n        k: target qubit.\n\n    Returns:\n        a unitary of n qubits with CNOT placed at ``j`` and ``k``.\n    '
    if j < k:
        unitary = np.kron(np.kron(np.eye(2 ** j), [[1, 0], [0, 0]]), np.eye(2 ** (n - 1 - j))) + np.kron(np.kron(np.kron(np.kron(np.eye(2 ** j), [[0, 0], [0, 1]]), np.eye(2 ** (k - j - 1))), [[0, 1], [1, 0]]), np.eye(2 ** (n - 1 - k)))
    else:
        unitary = np.kron(np.kron(np.eye(2 ** j), [[1, 0], [0, 0]]), np.eye(2 ** (n - 1 - j))) + np.kron(np.kron(np.kron(np.kron(np.eye(2 ** k), [[0, 1], [1, 0]]), np.eye(2 ** (j - k - 1))), [[0, 0], [0, 1]]), np.eye(2 ** (n - 1 - j)))
    return unitary

def rx_matrix(phi: float) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Computes an RX rotation by the angle of ``phi``.\n\n    Args:\n        phi: rotation angle.\n\n    Returns:\n        an RX rotation matrix.\n    '
    return RXGate(phi).to_matrix()

def ry_matrix(phi: float) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Computes an RY rotation by the angle of ``phi``.\n\n    Args:\n        phi: rotation angle.\n\n    Returns:\n        an RY rotation matrix.\n    '
    return RYGate(phi).to_matrix()

def rz_matrix(phi: float) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Computes an RZ rotation by the angle of ``phi``.\n\n    Args:\n        phi: rotation angle.\n\n    Returns:\n        an RZ rotation matrix.\n    '
    return RZGate(phi).to_matrix()