"""
Circuit synthesis for the Clifford class into layers.
"""
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.linear_phase.cx_cz_depth_lnn import synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix, _compute_rank, _gauss_elimination, _gauss_elimination_with_perm
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_h, _append_s, _append_cz

def _default_cx_synth_func(mat):
    if False:
        print('Hello World!')
    '\n    Construct the layer of CX gates from a boolean invertible matrix mat.\n    '
    CX_circ = synth_cnot_count_full_pmh(mat)
    CX_circ.name = 'CX'
    return CX_circ

def _default_cz_synth_func(symmetric_mat):
    if False:
        for i in range(10):
            print('nop')
    '\n    Construct the layer of CZ gates from a symmetric matrix.\n    '
    nq = symmetric_mat.shape[0]
    qc = QuantumCircuit(nq, name='CZ')
    for j in range(nq):
        for i in range(0, j):
            if symmetric_mat[i][j]:
                qc.cz(i, j)
    return qc

def synth_clifford_layers(cliff, cx_synth_func=_default_cx_synth_func, cz_synth_func=_default_cz_synth_func, cx_cz_synth_func=None, cz_func_reverse_qubits=False, validate=False):
    if False:
        print('Hello World!')
    'Synthesis of a Clifford into layers, it provides a similar decomposition to the synthesis\n    described in Lemma 8 of Bravyi and Maslov.\n\n    For example, a 5-qubit Clifford circuit is decomposed into the following layers:\n\n    .. parsed-literal::\n             ┌─────┐┌─────┐┌────────┐┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐\n        q_0: ┤0    ├┤0    ├┤0       ├┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├\n             │     ││     ││        ││     ││     ││     ││     ││        │\n        q_1: ┤1    ├┤1    ├┤1       ├┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├\n             │     ││     ││        ││     ││     ││     ││     ││        │\n        q_2: ┤2 S2 ├┤2 CZ ├┤2 CX_dg ├┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├\n             │     ││     ││        ││     ││     ││     ││     ││        │\n        q_3: ┤3    ├┤3    ├┤3       ├┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├\n             │     ││     ││        ││     ││     ││     ││     ││        │\n        q_4: ┤4    ├┤4    ├┤4       ├┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├\n             └─────┘└─────┘└────────┘└─────┘└─────┘└─────┘└─────┘└────────┘\n\n    This decomposition is for the default cz_synth_func and cx_synth_func functions,\n    with other functions one may see slightly different decomposition.\n\n    Args:\n        cliff (Clifford): a clifford operator.\n        cx_synth_func (Callable): a function to decompose the CX sub-circuit.\n            It gets as input a boolean invertible matrix, and outputs a QuantumCircuit.\n        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.\n            It gets as input a boolean symmetric matrix, and outputs a QuantumCircuit.\n        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.\n        validate (Boolean): if True, validates the synthesis process.\n        cz_func_reverse_qubits (Boolean): True only if cz_synth_func is synth_cz_depth_line_mr,\n            since this function returns a circuit that reverts the order of qubits.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the Clifford.\n\n    Reference:\n        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the\n           structure of the Clifford group*,\n           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_\n    '
    num_qubits = cliff.num_qubits
    if cz_func_reverse_qubits:
        cliff0 = _reverse_clifford(cliff)
    else:
        cliff0 = cliff
    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)
    (H1_circ, cliff1) = _create_graph_state(cliff0, validate=validate)
    (H2_circ, CZ1_circ, S1_circ, cliff2) = _decompose_graph_state(cliff1, validate=validate, cz_synth_func=cz_synth_func)
    (S2_circ, CZ2_circ, CX_circ) = _decompose_hadamard_free(cliff2.adjoint(), validate=validate, cz_synth_func=cz_synth_func, cx_synth_func=cx_synth_func, cx_cz_synth_func=cx_cz_synth_func, cz_func_reverse_qubits=cz_func_reverse_qubits)
    layeredCircuit.append(S2_circ, qubit_list)
    if cx_cz_synth_func is None:
        layeredCircuit.append(CZ2_circ, qubit_list)
        CXinv = CX_circ.copy().inverse()
        layeredCircuit.append(CXinv, qubit_list)
    else:
        layeredCircuit.append(CX_circ, qubit_list)
    layeredCircuit.append(H2_circ, qubit_list)
    layeredCircuit.append(S1_circ, qubit_list)
    layeredCircuit.append(CZ1_circ, qubit_list)
    if cz_func_reverse_qubits:
        H1_circ = H1_circ.reverse_bits()
    layeredCircuit.append(H1_circ, qubit_list)
    from qiskit.quantum_info.operators.symplectic import Clifford
    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list)
    return layeredCircuit

def _reverse_clifford(cliff):
    if False:
        print('Hello World!')
    'Reverse qubit order of a Clifford cliff'
    cliff_cpy = cliff.copy()
    cliff_cpy.stab_z = np.flip(cliff.stab_z, axis=1)
    cliff_cpy.destab_z = np.flip(cliff.destab_z, axis=1)
    cliff_cpy.stab_x = np.flip(cliff.stab_x, axis=1)
    cliff_cpy.destab_x = np.flip(cliff.destab_x, axis=1)
    return cliff_cpy

def _create_graph_state(cliff, validate=False):
    if False:
        for i in range(10):
            print('nop')
    'Given a Clifford cliff (denoted by U) that induces a stabilizer state U |0>,\n    apply a layer H1 of Hadamard gates to a subset of the qubits to make H1 U |0> into a graph state,\n    namely to make cliff.stab_x matrix have full rank.\n    Returns the QuantumCircuit H1_circ that includes the Hadamard gates and the updated Clifford\n    that induces the graph state.\n    The algorithm is based on Lemma 6 in [2].\n\n    Args:\n        cliff (Clifford): a clifford operator.\n        validate (Boolean): if True, validates the synthesis process.\n\n    Return:\n        H1_circ: a circuit containing a layer of Hadamard gates.\n        cliffh: cliffh.stab_x has full rank.\n\n    Raises:\n        QiskitError: if there are errors in the Gauss elimination process.\n\n    Reference:\n        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,\n           Phys. Rev. A 70, 052328 (2004).\n           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_\n    '
    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    H1_circ = QuantumCircuit(num_qubits, name='H1')
    cliffh = cliff.copy()
    if rank < num_qubits:
        stab = cliff.stab[:, :-1]
        stab = _gauss_elimination(stab, num_qubits)
        Cmat = stab[rank:num_qubits, num_qubits:]
        Cmat = np.transpose(Cmat)
        (Cmat, perm) = _gauss_elimination_with_perm(Cmat)
        perm = perm[0:num_qubits - rank]
        if validate:
            if _compute_rank(Cmat) != num_qubits - rank:
                raise QiskitError('The matrix Cmat after Gauss elimination has wrong rank.')
            if _compute_rank(stab[:, 0:num_qubits]) != rank:
                raise QiskitError('The matrix after Gauss elimination has wrong rank.')
            for i in range(rank, num_qubits):
                if stab[i, 0:num_qubits].any():
                    raise QiskitError('After Gauss elimination, the final num_qubits - rank rowscontain non-zero elements')
        for qubit in perm:
            H1_circ.h(qubit)
            _append_h(cliffh, qubit)
        if validate:
            stabh = cliffh.stab_x
            if _compute_rank(stabh) != num_qubits:
                raise QiskitError('The state is not a graph state.')
    return (H1_circ, cliffh)

def _decompose_graph_state(cliff, validate, cz_synth_func):
    if False:
        return 10
    'Assumes that a stabilizer state of the Clifford cliff (denoted by U) corresponds to a graph state.\n    Decompose it into the layers S1 - CZ1 - H2, such that:\n    S1 CZ1 H2 |0> = U |0>,\n    where S1_circ is a circuit that can contain only S gates,\n    CZ1_circ is a circuit that can contain only CZ gates, and\n    H2_circ is a circuit that can contain H gates on all qubits.\n\n    Args:\n        cliff (Clifford): a clifford operator corresponding to a graph state, cliff.stab_x has full rank.\n        validate (Boolean): if True, validates the synthesis process.\n        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.\n\n    Return:\n        S1_circ: a circuit that can contain only S gates.\n        CZ1_circ: a circuit that can contain only CZ gates.\n        H2_circ: a circuit containing a layer of Hadamard gates.\n        cliff_cpy: a Hadamard-free Clifford.\n\n    Raises:\n        QiskitError: if cliff does not induce a graph state.\n    '
    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    cliff_cpy = cliff.copy()
    if rank < num_qubits:
        raise QiskitError('The stabilizer state is not a graph state.')
    S1_circ = QuantumCircuit(num_qubits, name='S1')
    H2_circ = QuantumCircuit(num_qubits, name='H2')
    stabx = cliff.stab_x
    stabz = cliff.stab_z
    stabx_inv = calc_inverse_matrix(stabx, validate)
    stabz_update = np.matmul(stabx_inv, stabz) % 2
    if validate:
        if (stabz_update != stabz_update.T).any():
            raise QiskitError('The multiplication of stabx_inv and stab_z is not a symmetric matrix.')
    CZ1_circ = cz_synth_func(stabz_update)
    for j in range(num_qubits):
        for i in range(0, j):
            if stabz_update[i][j]:
                _append_cz(cliff_cpy, i, j)
    for i in range(0, num_qubits):
        if stabz_update[i][i]:
            S1_circ.s(i)
            _append_s(cliff_cpy, i)
    for qubit in range(num_qubits):
        H2_circ.h(qubit)
        _append_h(cliff_cpy, qubit)
    return (H2_circ, CZ1_circ, S1_circ, cliff_cpy)

def _decompose_hadamard_free(cliff, validate, cz_synth_func, cx_synth_func, cx_cz_synth_func, cz_func_reverse_qubits):
    if False:
        while True:
            i = 10
    'Assumes that the Clifford cliff is Hadamard free.\n    Decompose it into the layers S2 - CZ2 - CX, where\n    S2_circ is a circuit that can contain only S gates,\n    CZ2_circ is a circuit that can contain only CZ gates, and\n    CX_circ is a circuit that can contain CX gates.\n\n    Args:\n        cliff (Clifford): a Hadamard-free clifford operator.\n        validate (Boolean): if True, validates the synthesis process.\n        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.\n        cx_synth_func (Callable): a function to decompose the CX sub-circuit.\n        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.\n        cz_func_reverse_qubits (Boolean): True only if cz_synth_func is synth_cz_depth_line_mr.\n\n    Return:\n        S2_circ: a circuit that can contain only S gates.\n        CZ2_circ: a circuit that can contain only CZ gates.\n        CX_circ: a circuit that can contain only CX gates.\n\n    Raises:\n        QiskitError: if cliff is not Hadamard free.\n    '
    num_qubits = cliff.num_qubits
    destabx = cliff.destab_x
    destabz = cliff.destab_z
    stabx = cliff.stab_x
    if not (stabx == np.zeros((num_qubits, num_qubits))).all():
        raise QiskitError('The given Clifford is not Hadamard-free.')
    destabz_update = np.matmul(calc_inverse_matrix(destabx), destabz) % 2
    if validate:
        if (destabz_update != destabz_update.T).any():
            raise QiskitError('The multiplication of the inverse of destabx anddestabz is not a symmetric matrix.')
    S2_circ = QuantumCircuit(num_qubits, name='S2')
    for i in range(0, num_qubits):
        if destabz_update[i][i]:
            S2_circ.s(i)
    if cx_cz_synth_func is not None:
        for i in range(num_qubits):
            destabz_update[i][i] = 0
        mat_z = destabz_update
        mat_x = calc_inverse_matrix(destabx.transpose())
        CXCZ_circ = cx_cz_synth_func(mat_x, mat_z)
        return (S2_circ, QuantumCircuit(num_qubits), CXCZ_circ)
    CZ2_circ = cz_synth_func(destabz_update)
    mat = destabx.transpose()
    if cz_func_reverse_qubits:
        mat = np.flip(mat, axis=0)
    CX_circ = cx_synth_func(mat)
    return (S2_circ, CZ2_circ, CX_circ)

def _calc_pauli_diff(cliff, cliff_target):
    if False:
        print('Hello World!')
    'Given two Cliffords that differ by a Pauli, we find this Pauli.'
    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError('num_qubits is not the same for the original clifford and the target.')
    phase = [cliff.phase[k] ^ cliff_target.phase[k] for k in range(2 * num_qubits)]
    phase = np.array(phase, dtype=int)
    A = cliff.symplectic_matrix
    Ainv = calc_inverse_matrix(A)
    C = np.matmul(Ainv, phase) % 2
    pauli_circ = QuantumCircuit(num_qubits, name='Pauli')
    for k in range(num_qubits):
        destab = C[k]
        stab = C[k + num_qubits]
        if stab and destab:
            pauli_circ.y(k)
        elif stab:
            pauli_circ.x(k)
        elif destab:
            pauli_circ.z(k)
    return pauli_circ

def synth_clifford_depth_lnn(cliff):
    if False:
        for i in range(10):
            print('nop')
    'Synthesis of a Clifford into layers for linear-nearest neighbour connectivity.\n\n    The depth of the synthesized n-qubit circuit is bounded by 7*n+2, which is not optimal.\n    It should be replaced by a better algorithm that provides depth bounded by 7*n-4 [3].\n\n    Args:\n        cliff (Clifford): a clifford operator.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the Clifford.\n\n    Reference:\n        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the\n           structure of the Clifford group*,\n           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_\n        2. Dmitri Maslov, Martin Roetteler,\n           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,\n           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.\n        3. Dmitri Maslov, Willers Yang, *CNOT circuits need little help to implement arbitrary\n           Hadamard-free Clifford transformations they generate*,\n           `arXiv:2210.16195 <https://arxiv.org/abs/2210.16195>`_.\n    '
    circ = synth_clifford_layers(cliff, cx_synth_func=synth_cnot_depth_line_kms, cz_synth_func=synth_cz_depth_line_mr, cx_cz_synth_func=synth_cx_cz_depth_line_my, cz_func_reverse_qubits=True)
    return circ