"""
Circuit synthesis for the Clifford class.
"""
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_cx, _append_h, _append_s, _append_swap, _append_x, _append_z
from .clifford_decompose_bm import _decompose_clifford_1q

def synth_clifford_ag(clifford):
    if False:
        return 10
    'Decompose a Clifford operator into a QuantumCircuit based on Aaronson-Gottesman method.\n\n    Args:\n        clifford (Clifford): a clifford operator.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the Clifford.\n\n    Reference:\n        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,\n           Phys. Rev. A 70, 052328 (2004).\n           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_\n    '
    if clifford.num_qubits == 1:
        return _decompose_clifford_1q(clifford.tableau)
    circuit = QuantumCircuit(clifford.num_qubits, name=str(clifford))
    clifford_cpy = clifford.copy()
    for i in range(clifford.num_qubits):
        _set_qubit_x_true(clifford_cpy, circuit, i)
        _set_row_x_zero(clifford_cpy, circuit, i)
        _set_row_z_zero(clifford_cpy, circuit, i)
    for i in range(clifford.num_qubits):
        if clifford_cpy.destab_phase[i]:
            _append_z(clifford_cpy, i)
            circuit.z(i)
        if clifford_cpy.stab_phase[i]:
            _append_x(clifford_cpy, i)
            circuit.x(i)
    return circuit.inverse()

def _set_qubit_x_true(clifford, circuit, qubit):
    if False:
        return 10
    'Set destabilizer.X[qubit, qubit] to be True.\n\n    This is done by permuting columns l > qubit or if necessary applying\n    a Hadamard\n    '
    x = clifford.destab_x[qubit]
    z = clifford.destab_z[qubit]
    if x[qubit]:
        return
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_swap(clifford, i, qubit)
            circuit.swap(i, qubit)
            return
    for i in range(qubit, clifford.num_qubits):
        if z[i]:
            _append_h(clifford, i)
            circuit.h(i)
            if i != qubit:
                _append_swap(clifford, i, qubit)
                circuit.swap(i, qubit)
            return

def _set_row_x_zero(clifford, circuit, qubit):
    if False:
        print('Hello World!')
    'Set destabilizer.X[qubit, i] to False for all i > qubit.\n\n    This is done by applying CNOTS assumes k<=N and A[k][k]=1\n    '
    x = clifford.destab_x[qubit]
    z = clifford.destab_z[qubit]
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_cx(clifford, qubit, i)
            circuit.cx(qubit, i)
    if np.any(z[qubit:]):
        if not z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
        _append_s(clifford, qubit)
        circuit.s(qubit)

def _set_row_z_zero(clifford, circuit, qubit):
    if False:
        i = 10
        return i + 15
    'Set stabilizer.Z[qubit, i] to False for all i > qubit.\n\n    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits\n    and _set_row_x_zero has been called first\n    '
    x = clifford.stab_x[qubit]
    z = clifford.stab_z[qubit]
    if np.any(z[qubit + 1:]):
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
    if np.any(x[qubit:]):
        _append_h(clifford, qubit)
        circuit.h(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if x[i]:
                _append_cx(clifford, qubit, i)
                circuit.cx(qubit, i)
        if z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        _append_h(clifford, qubit)
        circuit.h(qubit)