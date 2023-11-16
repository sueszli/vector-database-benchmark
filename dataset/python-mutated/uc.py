"""Uniformly controlled gates (also called multiplexed gates)."""
from __future__ import annotations
import cmath
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from .diagonal import Diagonal
_EPS = 1e-10
_DECOMPOSER1Q = OneQubitEulerDecomposer('U3')

class UCGate(Gate):
    """Uniformly controlled gate (also called multiplexed gate).

    These gates can have several control qubits and a single target qubit.
    If the k control qubits are in the state :math:`|i\\rangle` (in the computational basis),
    a single-qubit unitary :math:`U_i` is applied to the target qubit.

    This gate is represented by a block-diagonal matrix, where each block is a
    :math:`2\\times 2` unitary, that is

    .. math::

        \\begin{pmatrix}
            U_0 & 0 & \\cdots & 0 \\\\
            0 & U_1 & \\cdots & 0 \\\\
            \\vdots  &     & \\ddots & \\vdots \\\\
            0 & 0   &  \\cdots & U_{2^{k-1}}
        \\end{pmatrix}.

    The decomposition is based on Ref. [1].

    **References:**

    [1] Bergholm et al., Quantum circuits with uniformly controlled one-qubit gates (2005).
        `Phys. Rev. A 71, 052330 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330>`__.

    """

    def __init__(self, gate_list: list[np.ndarray], up_to_diagonal: bool=False):
        if False:
            while True:
                i = 10
        "\n        Args:\n            gate_list: List of two qubit unitaries :math:`[U_0, ..., U_{2^{k-1}}]`, where each\n                single-qubit unitary :math:`U_i` is given as a :math:`2 \\times 2` numpy array.\n            up_to_diagonal: Determines if the gate is implemented up to a diagonal.\n                or if it is decomposed completely (default: False).\n                If the ``UCGate`` :math:`U` is decomposed up to a diagonal :math:`D`, this means\n                that the circuit implements a unitary :math:`U'` such that :math:`D U' = U`.\n\n        Raises:\n            QiskitError: in case of bad input to the constructor\n        "
        if not isinstance(gate_list, list):
            raise QiskitError('The single-qubit unitaries are not provided in a list.')
        for gate in gate_list:
            if not gate.shape == (2, 2):
                raise QiskitError('The dimension of a controlled gate is not equal to (2,2).')
        if not gate_list:
            raise QiskitError('The gate list cannot be empty.')
        num_contr = math.log2(len(gate_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError('The number of controlled single-qubit gates is not a non-negative power of 2.')
        for gate in gate_list:
            if not is_unitary_matrix(gate, _EPS):
                raise QiskitError('A controlled gate is not unitary.')
        super().__init__('multiplexer', int(num_contr) + 1, gate_list)
        self.up_to_diagonal = up_to_diagonal

    def inverse(self) -> Gate:
        if False:
            return 10
        'Return the inverse.\n\n        This does not re-compute the decomposition for the multiplexer with the inverse of the\n        gates but simply inverts the existing decomposition.\n        '
        inverse_gate = Gate(name=self.name + '_dg', num_qubits=self.num_qubits, params=[])
        definition = QuantumCircuit(*self.definition.qregs)
        for inst in reversed(self._definition):
            definition._append(inst.replace(operation=inst.operation.inverse()))
        definition.global_phase = -self.definition.global_phase
        inverse_gate.definition = definition
        return inverse_gate

    def _get_diagonal(self):
        if False:
            for i in range(10):
                print('nop')
        (_, diag) = self._dec_ucg()
        return diag

    def _define(self):
        if False:
            while True:
                i = 10
        (ucg_circuit, _) = self._dec_ucg()
        self.definition = ucg_circuit

    def _dec_ucg(self):
        if False:
            return 10
        '\n        Call to create a circuit that implements the uniformly controlled gate. If\n        up_to_diagonal=True, the circuit implements the gate up to a diagonal gate and\n        the diagonal gate is also returned.\n        '
        diag = np.ones(2 ** self.num_qubits).tolist()
        q = QuantumRegister(self.num_qubits)
        q_controls = q[1:]
        q_target = q[0]
        circuit = QuantumCircuit(q)
        if not q_controls:
            circuit.unitary(self.params[0], [q])
            return (circuit, diag)
        (single_qubit_gates, diag) = self._dec_ucg_help()
        for (i, gate) in enumerate(single_qubit_gates):
            if i == 0:
                squ = HGate().to_matrix().dot(gate)
            elif i == len(single_qubit_gates) - 1:
                squ = gate.dot(UCGate._rz(np.pi / 2)).dot(HGate().to_matrix())
            else:
                squ = HGate().to_matrix().dot(gate.dot(UCGate._rz(np.pi / 2))).dot(HGate().to_matrix())
            circuit.unitary(squ, [q_target])
            binary_rep = np.binary_repr(i + 1)
            num_trailing_zeros = len(binary_rep) - len(binary_rep.rstrip('0'))
            q_contr_index = num_trailing_zeros
            if not i == len(single_qubit_gates) - 1:
                circuit.cx(q_controls[q_contr_index], q_target)
                circuit.global_phase -= 0.25 * np.pi
        if not self.up_to_diagonal:
            diagonal = Diagonal(diag)
            circuit.append(diagonal, q)
        return (circuit, diag)

    def _dec_ucg_help(self):
        if False:
            i = 10
            return i + 15
        '\n        This method finds the single qubit gate arising in the decomposition of UCGates given in\n        https://arxiv.org/pdf/quant-ph/0410066.pdf.\n        '
        single_qubit_gates = [gate.astype(complex) for gate in self.params]
        diag = np.ones(2 ** self.num_qubits, dtype=complex)
        num_contr = self.num_qubits - 1
        for dec_step in range(num_contr):
            num_ucgs = 2 ** dec_step
            for ucg_index in range(num_ucgs):
                len_ucg = 2 ** (num_contr - dec_step)
                for i in range(int(len_ucg / 2)):
                    shift = ucg_index * len_ucg
                    a = single_qubit_gates[shift + i]
                    b = single_qubit_gates[shift + len_ucg // 2 + i]
                    (v, u, r) = self._demultiplex_single_uc(a, b)
                    single_qubit_gates[shift + i] = v
                    single_qubit_gates[shift + len_ucg // 2 + i] = u
                    if ucg_index < num_ucgs - 1:
                        k = shift + len_ucg + i
                        single_qubit_gates[k] = single_qubit_gates[k].dot(UCGate._ct(r)) * UCGate._rz(np.pi / 2).item((0, 0))
                        k = k + len_ucg // 2
                        single_qubit_gates[k] = single_qubit_gates[k].dot(r) * UCGate._rz(np.pi / 2).item((1, 1))
                    else:
                        for ucg_index_2 in range(num_ucgs):
                            shift_2 = ucg_index_2 * len_ucg
                            k = 2 * (i + shift_2)
                            diag[k] = diag[k] * UCGate._ct(r).item((0, 0)) * UCGate._rz(np.pi / 2).item((0, 0))
                            diag[k + 1] = diag[k + 1] * UCGate._ct(r).item((1, 1)) * UCGate._rz(np.pi / 2).item((0, 0))
                            k = len_ucg + k
                            diag[k] *= r.item((0, 0)) * UCGate._rz(np.pi / 2).item((1, 1))
                            diag[k + 1] *= r.item((1, 1)) * UCGate._rz(np.pi / 2).item((1, 1))
        return (single_qubit_gates, diag)

    def _demultiplex_single_uc(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method implements the decomposition given in equation (3) in\n        https://arxiv.org/pdf/quant-ph/0410066.pdf.\n        The decomposition is used recursively to decompose uniformly controlled gates.\n        a,b = single qubit unitaries\n        v,u,r = outcome of the decomposition given in the reference mentioned above\n        (see there for the details).\n        '
        x = a.dot(UCGate._ct(b))
        det_x = np.linalg.det(x)
        x11 = x.item((0, 0)) / cmath.sqrt(det_x)
        phi = cmath.phase(det_x)
        r1 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 - cmath.phase(x11)))
        r2 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 + cmath.phase(x11) + np.pi))
        r = np.array([[r1, 0], [0, r2]], dtype=complex)
        (d, u) = np.linalg.eig(r.dot(x).dot(r))
        if abs(d[0] + 1j) < _EPS:
            d = np.flip(d, 0)
            u = np.flip(u, 1)
        d = np.diag(np.sqrt(d))
        v = d.dot(UCGate._ct(u)).dot(UCGate._ct(r)).dot(b)
        return (v, u, r)

    @staticmethod
    def _ct(m):
        if False:
            i = 10
            return i + 15
        return np.transpose(np.conjugate(m))

    @staticmethod
    def _rz(alpha):
        if False:
            while True:
                i = 10
        return np.array([[np.exp(1j * alpha / 2), 0], [0, np.exp(-1j * alpha / 2)]])

    def validate_parameter(self, parameter):
        if False:
            i = 10
            return i + 15
        'Uniformly controlled gate parameter has to be an ndarray.'
        if isinstance(parameter, np.ndarray):
            return parameter
        else:
            raise CircuitError(f'invalid param type {type(parameter)} in gate {self.name}')