"""Contains functions used by the basic aer simulators.

"""
from string import ascii_uppercase, ascii_lowercase
from typing import List, Optional
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.exceptions import QiskitError
SINGLE_QUBIT_GATES = ('U', 'u', 'h', 'p', 'u1', 'u2', 'u3', 'rz', 'sx', 'x')

def single_gate_matrix(gate: str, params: Optional[List[float]]=None):
    if False:
        print('Hello World!')
    "Get the matrix for a single qubit.\n\n    Args:\n        gate: the single qubit gate name\n        params: the operation parameters op['params']\n    Returns:\n        array: A numpy array representing the matrix\n    Raises:\n        QiskitError: If a gate outside the supported set is passed in for the\n            ``Gate`` argument.\n    "
    if params is None:
        params = []
    if gate == 'U':
        gc = gates.UGate
    elif gate == 'u3':
        gc = gates.U3Gate
    elif gate == 'h':
        gc = gates.HGate
    elif gate == 'u':
        gc = gates.UGate
    elif gate == 'p':
        gc = gates.PhaseGate
    elif gate == 'u2':
        gc = gates.U2Gate
    elif gate == 'u1':
        gc = gates.U1Gate
    elif gate == 'rz':
        gc = gates.RZGate
    elif gate == 'id':
        gc = gates.IGate
    elif gate == 'sx':
        gc = gates.SXGate
    elif gate == 'x':
        gc = gates.XGate
    else:
        raise QiskitError('Gate is not a valid basis gate for this simulator: %s' % gate)
    return gc(*params).to_matrix()
_CX_MATRIX = gates.CXGate().to_matrix()

def cx_gate_matrix():
    if False:
        return 10
    'Get the matrix for a controlled-NOT gate.'
    return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)

def einsum_matmul_index(gate_indices, number_of_qubits):
    if False:
        print('Hello World!')
    'Return the index string for Numpy.einsum matrix-matrix multiplication.\n\n    The returned indices are to perform a matrix multiplication A.B where\n    the matrix A is an M-qubit matrix, matrix B is an N-qubit matrix, and\n    M <= N, and identity matrices are implied on the subsystems where A has no\n    support on B.\n\n    Args:\n        gate_indices (list[int]): the indices of the right matrix subsystems\n                                   to contract with the left matrix.\n        number_of_qubits (int): the total number of qubits for the right matrix.\n\n    Returns:\n        str: An indices string for the Numpy.einsum function.\n    '
    (mat_l, mat_r, tens_lin, tens_lout) = _einsum_matmul_index_helper(gate_indices, number_of_qubits)
    tens_r = ascii_uppercase[:number_of_qubits]
    return '{mat_l}{mat_r}, '.format(mat_l=mat_l, mat_r=mat_r) + '{tens_lin}{tens_r}->{tens_lout}{tens_r}'.format(tens_lin=tens_lin, tens_lout=tens_lout, tens_r=tens_r)

def einsum_vecmul_index(gate_indices, number_of_qubits):
    if False:
        while True:
            i = 10
    'Return the index string for Numpy.einsum matrix-vector multiplication.\n\n    The returned indices are to perform a matrix multiplication A.v where\n    the matrix A is an M-qubit matrix, vector v is an N-qubit vector, and\n    M <= N, and identity matrices are implied on the subsystems where A has no\n    support on v.\n\n    Args:\n        gate_indices (list[int]): the indices of the right matrix subsystems\n                                  to contract with the left matrix.\n        number_of_qubits (int): the total number of qubits for the right matrix.\n\n    Returns:\n        str: An indices string for the Numpy.einsum function.\n    '
    (mat_l, mat_r, tens_lin, tens_lout) = _einsum_matmul_index_helper(gate_indices, number_of_qubits)
    return f'{mat_l}{mat_r}, ' + '{tens_lin}->{tens_lout}'.format(tens_lin=tens_lin, tens_lout=tens_lout)

def _einsum_matmul_index_helper(gate_indices, number_of_qubits):
    if False:
        for i in range(10):
            print('nop')
    'Return the index string for Numpy.einsum matrix multiplication.\n\n    The returned indices are to perform a matrix multiplication A.v where\n    the matrix A is an M-qubit matrix, matrix v is an N-qubit vector, and\n    M <= N, and identity matrices are implied on the subsystems where A has no\n    support on v.\n\n    Args:\n        gate_indices (list[int]): the indices of the right matrix subsystems\n                                   to contract with the left matrix.\n        number_of_qubits (int): the total number of qubits for the right matrix.\n\n    Returns:\n        tuple: (mat_left, mat_right, tens_in, tens_out) of index strings for\n        that may be combined into a Numpy.einsum function string.\n\n    Raises:\n        QiskitError: if the total number of qubits plus the number of\n        contracted indices is greater than 26.\n    '
    if len(gate_indices) + number_of_qubits > 26:
        raise QiskitError('Total number of free indexes limited to 26')
    tens_in = ascii_lowercase[:number_of_qubits]
    tens_out = list(tens_in)
    mat_left = ''
    mat_right = ''
    for (pos, idx) in enumerate(reversed(gate_indices)):
        mat_left += ascii_lowercase[-1 - pos]
        mat_right += tens_in[-1 - idx]
        tens_out[-1 - idx] = ascii_lowercase[-1 - pos]
    tens_out = ''.join(tens_out)
    return (mat_left, mat_right, tens_in, tens_out)