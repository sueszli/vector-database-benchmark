"""
Quantum Shannon Decomposition.

Method is described in arXiv:quant-ph/0406176.
"""
from __future__ import annotations
import scipy
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.synthesis import two_qubit_decompose, one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit.circuit.library.generalized_gates.ucry import UCRYGate
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate

def qs_decomposition(mat: np.ndarray, opt_a1: bool=True, opt_a2: bool=True, decomposer_1q=None, decomposer_2q=None, *, _depth=0):
    if False:
        i = 10
        return i + 15
    '\n    Decomposes unitary matrix into one and two qubit gates using Quantum Shannon Decomposition.\n\n       ┌───┐               ┌───┐     ┌───┐     ┌───┐\n      ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────\n       │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐\n     /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├\n       └───┘          └───┘     └───┘     └───┘     └───┘\n\n    The number of CX gates generated with the decomposition without optimizations is,\n\n    .. math::\n\n        \x0crac{9}{16} 4^n - frac{3}{2} 2^n\n\n    If opt_a1 = True, the default, the CX count is reduced by,\n\n    .. math::\n\n        \x0crac{1}{3} 4^{n - 2} - 1.\n\n    If opt_a2 = True, the default, the CX count is reduced by,\n\n    .. math::\n\n        4^{n-2} - 1.\n\n    This decomposition is described in arXiv:quant-ph/0406176.\n\n    Arguments:\n       mat (ndarray): unitary matrix to decompose\n       opt_a1 (bool): whether to try optimization A.1 from Shende. This should eliminate 1 cnot\n          per call. If True CZ gates are left in the output. If desired these can be further decomposed\n          to CX.\n       opt_a2 (bool): whether to try optimization A.2 from Shende. This decomposes two qubit\n          unitaries into a diagonal gate and a two cx unitary and reduces overal cx count by\n          4^(n-2) - 1.\n       decomposer_1q (None or Object): optional 1Q decomposer. If None, uses\n          :class:`~qiskit.quantum_info.synthesis.one_qubit_decomposer.OneQubitEulerDecomser`\n       decomposer_2q (None or Object): optional 2Q decomposer. If None, uses\n          :class:`~qiskit.quantum_info.synthesis.two_qubit_decomposer.two_qubit_cnot_decompose\n\n    Return:\n       QuantumCircuit: Decomposed quantum circuit.\n    '
    dim = mat.shape[0]
    nqubits = int(np.log2(dim))
    if np.allclose(np.identity(dim), mat):
        return QuantumCircuit(nqubits)
    if dim == 2:
        if decomposer_1q is None:
            decomposer_1q = one_qubit_decompose.OneQubitEulerDecomposer()
        circ = decomposer_1q(mat)
    elif dim == 4:
        if decomposer_2q is None:
            if opt_a2 and _depth > 0:
                from qiskit.circuit.library import UnitaryGate

                def decomp_2q(mat):
                    if False:
                        print('Hello World!')
                    ugate = UnitaryGate(mat)
                    qc = QuantumCircuit(2, name='qsd2q')
                    qc.append(ugate, [0, 1])
                    return qc
                decomposer_2q = decomp_2q
            else:
                decomposer_2q = two_qubit_decompose.two_qubit_cnot_decompose
        circ = decomposer_2q(mat)
    else:
        qr = QuantumRegister(nqubits)
        circ = QuantumCircuit(qr)
        dim_o2 = dim // 2
        ((u1, u2), vtheta, (v1h, v2h)) = scipy.linalg.cossin(mat, separate=True, p=dim_o2, q=dim_o2)
        left_circ = _demultiplex(v1h, v2h, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth)
        circ.append(left_circ.to_instruction(), qr)
        if opt_a1:
            nangles = len(vtheta)
            half_size = nangles // 2
            circ_cz = _get_ucry_cz(nqubits, (2 * vtheta).tolist())
            circ.append(circ_cz.to_instruction(), range(nqubits))
            u2[:, half_size:] = np.negative(u2[:, half_size:])
        else:
            ucry = UCRYGate((2 * vtheta).tolist())
            circ.append(ucry, [qr[-1]] + qr[:-1])
        right_circ = _demultiplex(u1, u2, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth)
        circ.append(right_circ.to_instruction(), qr)
    if opt_a2 and _depth == 0 and (dim > 4):
        return _apply_a2(circ)
    return circ

def _demultiplex(um0, um1, opt_a1=False, opt_a2=False, *, _depth=0):
    if False:
        print('Hello World!')
    'Decompose a generic multiplexer.\n\n          ────□────\n           ┌──┴──┐\n         /─┤     ├─\n           └─────┘\n\n    represented by the block diagonal matrix\n\n            ┏         ┓\n            ┃ um0     ┃\n            ┃     um1 ┃\n            ┗         ┛\n\n    to\n               ┌───┐\n        ───────┤ Rz├──────\n          ┌───┐└─┬─┘┌───┐\n        /─┤ w ├──□──┤ v ├─\n          └───┘     └───┘\n\n    where v and w are general unitaries determined from decomposition.\n\n    Args:\n       um0 (ndarray): applied if MSB is 0\n       um1 (ndarray): applied if MSB is 1\n       opt_a1 (bool): whether to try optimization A.1 from Shende. This should elliminate 1 cnot\n          per call. If True CZ gates are left in the output. If desired these can be further decomposed\n       opt_a2 (bool): whether to try  optimization A.2 from Shende. This decomposes two qubit\n          unitaries into a diagonal gate and a two cx unitary and reduces overal cx count by\n          4^(n-2) - 1.\n       _depth (int): This is an internal variable to track the recursion depth.\n\n    Returns:\n        QuantumCircuit: decomposed circuit\n    '
    dim = um0.shape[0] + um1.shape[0]
    nqubits = int(np.log2(dim))
    um0um1 = um0 @ um1.T.conjugate()
    if is_hermitian_matrix(um0um1):
        (eigvals, vmat) = scipy.linalg.eigh(um0um1)
    else:
        (evals, vmat) = scipy.linalg.schur(um0um1, output='complex')
        eigvals = evals.diagonal()
    dvals = np.emath.sqrt(eigvals)
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ um1
    circ = QuantumCircuit(nqubits)
    left_gate = qs_decomposition(wmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1).to_instruction()
    circ.append(left_gate, range(nqubits - 1))
    angles = 2 * np.angle(np.conj(dvals))
    ucrz = UCRZGate(angles.tolist())
    circ.append(ucrz, [nqubits - 1] + list(range(nqubits - 1)))
    right_gate = qs_decomposition(vmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1).to_instruction()
    circ.append(right_gate, range(nqubits - 1))
    return circ

def _get_ucry_cz(nqubits, angles):
    if False:
        i = 10
        return i + 15
    '\n    Get uniformly controlled Ry gate in in CZ-Ry as in UCPauliRotGate.\n    '
    nangles = len(angles)
    qc = QuantumCircuit(nqubits)
    q_controls = qc.qubits[:-1]
    q_target = qc.qubits[-1]
    if not q_controls:
        if np.abs(angles[0]) > _EPS:
            qc.ry(angles[0], q_target)
    else:
        angles = angles.copy()
        UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
        for (i, angle) in enumerate(angles):
            if np.abs(angle) > _EPS:
                qc.ry(angle, q_target)
            if not i == len(angles) - 1:
                binary_rep = np.binary_repr(i + 1)
                q_contr_index = len(binary_rep) - len(binary_rep.rstrip('0'))
            else:
                q_contr_index = len(q_controls) - 1
            if i < nangles - 1:
                qc.cz(q_controls[q_contr_index], q_target)
    return qc

def _apply_a2(circ):
    if False:
        return 10
    from qiskit import transpile
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library.generalized_gates import UnitaryGate
    decomposer = two_qubit_decompose.TwoQubitDecomposeUpToDiagonal()
    ccirc = transpile(circ, basis_gates=['u', 'cx', 'qsd2q'], optimization_level=0)
    ind2q = []
    for (i, instruction) in enumerate(ccirc.data):
        if instruction.operation.name == 'qsd2q':
            ind2q.append(i)
    if len(ind2q) == 0:
        return ccirc
    elif len(ind2q) == 1:
        ccirc.data[ind2q[0]].operation.name = 'Unitary'
        return ccirc
    ind2 = None
    for (ind1, ind2) in zip(ind2q[0:-1], ind2q[1:]):
        instr1 = ccirc.data[ind1]
        mat1 = Operator(instr1.operation).data
        instr2 = ccirc.data[ind2]
        mat2 = Operator(instr2.operation).data
        (dmat, qc2cx) = decomposer(mat1)
        ccirc.data[ind1] = instr1.replace(operation=qc2cx.to_gate())
        mat2 = mat2 @ dmat
        ccirc.data[ind2] = instr2.replace(UnitaryGate(mat2))
    qc3 = two_qubit_decompose.two_qubit_cnot_decompose(mat2)
    ccirc.data[ind2] = ccirc.data[ind2].replace(operation=qc3.to_gate())
    return ccirc