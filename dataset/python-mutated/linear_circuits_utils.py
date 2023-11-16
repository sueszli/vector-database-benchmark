"""Utility functions for handling linear reversible circuits."""
import copy
from typing import Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from . import calc_inverse_matrix, check_invertible_binary_matrix

def transpose_cx_circ(qc: QuantumCircuit):
    if False:
        while True:
            i = 10
    'Takes a circuit having only CX gates, and calculates its transpose.\n    This is done by recursively replacing CX(i, j) with CX(j, i) in all instructions.\n\n    Args:\n        qc: a QuantumCircuit containing only CX gates.\n\n    Returns:\n        QuantumCircuit: the transposed circuit.\n\n    Raises:\n        CircuitError: if qc has a non-CX gate.\n    '
    transposed_circ = QuantumCircuit(qc.qubits, qc.clbits, name=qc.name + '_transpose')
    for instruction in reversed(qc.data):
        if instruction.operation.name != 'cx':
            raise CircuitError('The circuit contains non-CX gates.')
        transposed_circ._append(instruction.replace(qubits=reversed(instruction.qubits)))
    return transposed_circ

def optimize_cx_4_options(function: Callable, mat: np.ndarray, optimize_count: bool=True):
    if False:
        print('Hello World!')
    'Get the best implementation of a circuit implementing a binary invertible matrix M,\n    by considering all four options: M,M^(-1),M^T,M^(-1)^T.\n    Optimizing either the CX count or the depth.\n\n    Args:\n        function: the synthesis function.\n        mat: a binary invertible matrix.\n        optimize_count: True if the number of CX gates in optimize, False if the depth is optimized.\n\n    Returns:\n        QuantumCircuit: an optimized QuantumCircuit, has the best depth or CX count of the four options.\n\n    Raises:\n        QiskitError: if mat is not an invertible matrix.\n    '
    if not check_invertible_binary_matrix(mat):
        raise QiskitError('The matrix is not invertible.')
    qc = function(mat)
    best_qc = qc
    best_depth = qc.depth()
    best_count = qc.count_ops()['cx']
    for i in range(1, 4):
        mat_cpy = copy.deepcopy(mat)
        if i == 1:
            mat_cpy = calc_inverse_matrix(mat_cpy)
            qc = function(mat_cpy)
            qc = qc.inverse()
        elif i == 2:
            mat_cpy = np.transpose(mat_cpy)
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
        elif i == 3:
            mat_cpy = calc_inverse_matrix(np.transpose(mat_cpy))
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
            qc = qc.inverse()
        new_depth = qc.depth()
        new_count = qc.count_ops()['cx']
        better_count = optimize_count and best_count > new_count or (not optimize_count and best_depth == new_depth and (best_count > new_count))
        better_depth = not optimize_count and best_depth > new_depth or (optimize_count and best_count == new_count and (best_depth > new_depth))
        if better_count or better_depth:
            best_count = new_count
            best_depth = new_depth
            best_qc = qc
    return best_qc

def check_lnn_connectivity(qc: QuantumCircuit) -> bool:
    if False:
        while True:
            i = 10
    'Check that the synthesized circuit qc fits linear nearest neighbor connectivity.\n\n    Args:\n        qc: a QuantumCircuit containing only CX and single qubit gates.\n\n    Returns:\n        bool: True if the circuit has linear nearest neighbor connectivity.\n\n    Raises:\n        CircuitError: if qc has a non-CX two-qubit gate.\n    '
    for instruction in qc.data:
        if instruction.operation.num_qubits > 1:
            if instruction.operation.name == 'cx':
                q0 = qc.find_bit(instruction.qubits[0]).index
                q1 = qc.find_bit(instruction.qubits[1]).index
                dist = abs(q0 - q1)
                if dist != 1:
                    return False
            else:
                raise CircuitError('The circuit has two-qubits gates different than CX.')
    return True