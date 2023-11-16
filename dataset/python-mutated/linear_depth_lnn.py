"""
Optimize the synthesis of an n-qubit circuit contains only CX gates for
linear nearest neighbor (LNN) connectivity.
The depth of the circuit is bounded by 5*n, while the gate count is approximately 2.5*n^2

References:
    [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
         Computation at a Distance.
         `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
"""
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix, check_invertible_binary_matrix, _col_op, _row_op

def _row_op_update_instructions(cx_instructions, mat, a, b):
    if False:
        i = 10
        return i + 15
    cx_instructions.append((a, b))
    _row_op(mat, a, b)

def _get_lower_triangular(n, mat, mat_inv):
    if False:
        i = 10
        return i + 15
    mat = mat.copy()
    mat_t = mat.copy()
    mat_inv_t = mat_inv.copy()
    cx_instructions_rows = []
    for i in reversed(range(0, n)):
        found_first = False
        for j in reversed(range(0, n)):
            if mat[i, j]:
                if not found_first:
                    found_first = True
                    first_j = j
                else:
                    _col_op(mat, j, first_j)
        for k in reversed(range(0, i)):
            if mat[k, first_j]:
                _row_op_update_instructions(cx_instructions_rows, mat, i, k)
    for inst in cx_instructions_rows:
        _row_op(mat_t, inst[0], inst[1])
        _col_op(mat_inv_t, inst[0], inst[1])
    return (mat_t, mat_inv_t)

def _get_label_arr(n, mat_t):
    if False:
        print('Hello World!')
    label_arr = []
    for i in range(n):
        j = 0
        while not mat_t[i, n - 1 - j]:
            j += 1
        label_arr.append(j)
    return label_arr

def _in_linear_combination(label_arr_t, mat_inv_t, row, k):
    if False:
        i = 10
        return i + 15
    indx_k = label_arr_t[k]
    w_needed = np.zeros(len(row), dtype=bool)
    for (row_l, _) in enumerate(row):
        if row[row_l]:
            w_needed = w_needed ^ mat_inv_t[row_l]
    if w_needed[indx_k]:
        return False
    return True

def _get_label_arr_t(n, label_arr):
    if False:
        print('Hello World!')
    label_arr_t = [None] * n
    for i in range(n):
        label_arr_t[label_arr[i]] = i
    return label_arr_t

def _matrix_to_north_west(n, mat, mat_inv):
    if False:
        while True:
            i = 10
    (mat_t, mat_inv_t) = _get_lower_triangular(n, mat, mat_inv)
    label_arr = _get_label_arr(n, mat_t)
    label_arr_t = _get_label_arr_t(n, label_arr)
    first_qubit = 0
    empty_layers = 0
    done = False
    cx_instructions_rows = []
    while not done:
        at_least_one_needed = False
        for i in range(first_qubit, n - 1, 2):
            if label_arr[i] > label_arr[i + 1]:
                at_least_one_needed = True
                if _in_linear_combination(label_arr_t, mat_inv_t, mat[i + 1], label_arr[i + 1]):
                    pass
                elif _in_linear_combination(label_arr_t, mat_inv_t, mat[i + 1] ^ mat[i], label_arr[i + 1]):
                    _row_op_update_instructions(cx_instructions_rows, mat, i, i + 1)
                elif _in_linear_combination(label_arr_t, mat_inv_t, mat[i], label_arr[i + 1]):
                    _row_op_update_instructions(cx_instructions_rows, mat, i + 1, i)
                    _row_op_update_instructions(cx_instructions_rows, mat, i, i + 1)
                (label_arr[i], label_arr[i + 1]) = (label_arr[i + 1], label_arr[i])
        if not at_least_one_needed:
            empty_layers += 1
            if empty_layers > 1:
                done = True
        else:
            empty_layers = 0
        first_qubit = int(not first_qubit)
    return cx_instructions_rows

def _north_west_to_identity(n, mat):
    if False:
        print('Hello World!')
    label_arr = list(reversed(range(n)))
    first_qubit = 0
    empty_layers = 0
    done = False
    cx_instructions_rows = []
    while not done:
        at_least_one_needed = False
        for i in range(first_qubit, n - 1, 2):
            if label_arr[i] > label_arr[i + 1]:
                at_least_one_needed = True
                if not mat[i, label_arr[i + 1]]:
                    _row_op_update_instructions(cx_instructions_rows, mat, i + 1, i)
                _row_op_update_instructions(cx_instructions_rows, mat, i, i + 1)
                _row_op_update_instructions(cx_instructions_rows, mat, i + 1, i)
                (label_arr[i], label_arr[i + 1]) = (label_arr[i + 1], label_arr[i])
        if not at_least_one_needed:
            empty_layers += 1
            if empty_layers > 1:
                done = True
        else:
            empty_layers = 0
        first_qubit = int(not first_qubit)
    return cx_instructions_rows

def _optimize_cx_circ_depth_5n_line(mat):
    if False:
        print('Hello World!')
    mat_inv = mat.copy()
    mat_cpy = calc_inverse_matrix(mat_inv)
    n = len(mat_cpy)
    cx_instructions_rows_m2nw = _matrix_to_north_west(n, mat_cpy, mat_inv)
    cx_instructions_rows_nw2id = _north_west_to_identity(n, mat_cpy)
    return (cx_instructions_rows_m2nw, cx_instructions_rows_nw2id)

def synth_cnot_depth_line_kms(mat):
    if False:
        return 10
    '\n    Synthesize linear reversible circuit for linear nearest-neighbor architectures using\n    Kutin, Moulton, Smithline method.\n\n    Synthesis algorithm for linear reversible circuits from [1], Chapter 7.\n    Synthesizes any linear reversible circuit of n qubits over linear nearest-neighbor\n    architecture using CX gates with depth at most 5*n.\n\n    Args:\n        mat(np.ndarray]): A boolean invertible matrix.\n\n    Returns:\n        QuantumCircuit: the synthesized quantum circuit.\n\n    Raises:\n        QiskitError: if mat is not invertible.\n\n    References:\n        1. Kutin, S., Moulton, D. P., Smithline, L.,\n           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),\n           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_\n    '
    if not check_invertible_binary_matrix(mat):
        raise QiskitError('The input matrix is not invertible.')
    num_qubits = len(mat)
    cx_inst = _optimize_cx_circ_depth_5n_line(mat)
    qc = QuantumCircuit(num_qubits)
    for pair in cx_inst[0]:
        qc.cx(pair[0], pair[1])
    for pair in cx_inst[1]:
        qc.cx(pair[0], pair[1])
    return qc