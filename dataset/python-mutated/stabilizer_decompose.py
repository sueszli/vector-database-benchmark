"""
Circuit synthesis for a stabilizer state preparation circuit.
"""
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import StabilizerState
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.clifford.clifford_decompose_layers import _default_cz_synth_func, _reverse_clifford, _create_graph_state, _decompose_graph_state

def synth_stabilizer_layers(stab, cz_synth_func=_default_cz_synth_func, cz_func_reverse_qubits=False, validate=False):
    if False:
        i = 10
        return i + 15
    'Synthesis of a stabilizer state into layers.\n\n    It provides a similar decomposition to the synthesis described in Lemma 8 of Bravyi and Maslov,\n    without the initial Hadamard-free sub-circuit which do not affect the stabilizer state.\n\n    For example, a 5-qubit stabilizer state is decomposed into the following layers:\n\n    .. parsed-literal::\n             ┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐\n        q_0: ┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├\n             │     ││     ││     ││     ││        │\n        q_1: ┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├\n             │     ││     ││     ││     ││        │\n        q_2: ┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├\n             │     ││     ││     ││     ││        │\n        q_3: ┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├\n             │     ││     ││     ││     ││        │\n        q_4: ┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├\n             └─────┘└─────┘└─────┘└─────┘└────────┘\n\n    Args:\n        stab (StabilizerState): a stabilizer state.\n        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.\n            It gets as input a boolean symmetric matrix, and outputs a QuantumCircuit.\n        validate (Boolean): if True, validates the synthesis process.\n        cz_func_reverse_qubits (Boolean): True only if cz_synth_func is synth_cz_depth_line_mr,\n            since this function returns a circuit that reverts the order of qubits.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the stabilizer state.\n\n    Raises:\n        QiskitError: if the input is not a StabilizerState.\n\n    Reference:\n        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the\n           structure of the Clifford group*,\n           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_\n    '
    if not isinstance(stab, StabilizerState):
        raise QiskitError('The input is not a StabilizerState.')
    cliff = stab.clifford
    num_qubits = cliff.num_qubits
    if cz_func_reverse_qubits:
        cliff0 = _reverse_clifford(cliff)
    else:
        cliff0 = cliff
    (H1_circ, cliff1) = _create_graph_state(cliff0, validate=validate)
    (H2_circ, CZ1_circ, S1_circ, _) = _decompose_graph_state(cliff1, validate=validate, cz_synth_func=cz_synth_func)
    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)
    layeredCircuit.append(H2_circ, qubit_list)
    layeredCircuit.append(S1_circ, qubit_list)
    layeredCircuit.append(CZ1_circ, qubit_list)
    if cz_func_reverse_qubits:
        H1_circ = H1_circ.reverse_bits()
    layeredCircuit.append(H1_circ, qubit_list)
    from qiskit.quantum_info.operators.symplectic import Clifford
    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff_stabilizer(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list)
    return layeredCircuit

def _calc_pauli_diff_stabilizer(cliff, cliff_target):
    if False:
        print('Hello World!')
    'Given two Cliffords whose stabilizers differ by a Pauli, we find this Pauli.'
    from qiskit.quantum_info.operators.symplectic import Pauli
    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError('num_qubits is not the same for the original clifford and the target.')
    stab_gen = StabilizerState(cliff).clifford.to_dict()['stabilizer']
    ts = StabilizerState(cliff_target)
    phase_destab = [False] * num_qubits
    phase_stab = [ts.expectation_value(Pauli(stab_gen[i])) == -1 for i in range(num_qubits)]
    phase = []
    phase.extend(phase_destab)
    phase.extend(phase_stab)
    phase = np.array(phase, dtype=int)
    A = cliff.symplectic_matrix.astype(int)
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

def synth_stabilizer_depth_lnn(stab):
    if False:
        print('Hello World!')
    'Synthesis of an n-qubit stabilizer state for linear-nearest neighbour connectivity,\n    in 2-qubit depth 2*n+2 and two distinct CX layers, using CX and phase gates (S, Sdg or Z).\n\n    Args:\n        stab (StabilizerState): a stabilizer state.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the stabilizer state.\n\n    Reference:\n        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the\n           structure of the Clifford group*,\n           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_\n        2. Dmitri Maslov, Martin Roetteler,\n           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,\n           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.\n    '
    circ = synth_stabilizer_layers(stab, cz_synth_func=synth_cz_depth_line_mr, cz_func_reverse_qubits=True)
    return circ