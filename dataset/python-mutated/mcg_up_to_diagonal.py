"""Multi controlled single-qubit unitary up to diagonal."""
import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .uc import UCGate
_EPS = 1e-10

class MCGupDiag(Gate):
    """
    Decomposes a multi-controlled gate :math:`U` up to a diagonal :math:`D` acting on the control
    and target qubit (but not on the ancilla qubits), i.e., it implements a circuit corresponding to
    a unitary :math:`U'`, such that :math:`U = D U'`.
    """

    def __init__(self, gate: np.ndarray, num_controls: int, num_ancillas_zero: int, num_ancillas_dirty: int) -> None:
        if False:
            return 10
        '\n        Args:\n            gate: :math:`2 \\times 2` unitary given as a (complex) ``ndarray``.\n            num_controls: Number of control qubits.\n            num_ancillas_zero: Number of ancilla qubits that start in the state zero.\n            num_ancillas_dirty: Number of ancilla qubits that are allowed to start in an\n                arbitrary state.\n\n        Raises:\n            QiskitError: if the input format is wrong; if the array gate is not unitary\n        '
        self.num_controls = num_controls
        self.num_ancillas_zero = num_ancillas_zero
        self.num_ancillas_dirty = num_ancillas_dirty
        if not gate.shape == (2, 2):
            raise QiskitError('The dimension of the controlled gate is not equal to (2,2).')
        if not is_isometry(gate, _EPS):
            raise QiskitError('The controlled gate is not unitary.')
        num_qubits = 1 + num_controls + num_ancillas_zero + num_ancillas_dirty
        super().__init__('MCGupDiag', num_qubits, [gate])

    def _define(self):
        if False:
            i = 10
            return i + 15
        (mcg_up_diag_circuit, _) = self._dec_mcg_up_diag()
        gate = mcg_up_diag_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        mcg_up_diag_circuit = QuantumCircuit(q)
        mcg_up_diag_circuit.append(gate, q[:])
        self.definition = mcg_up_diag_circuit

    def inverse(self) -> Gate:
        if False:
            for i in range(10):
                print('nop')
        'Return the inverse.\n\n        Note that the resulting Gate object has an empty ``params`` property.\n        '
        inverse_gate = Gate(name=self.name + '_dg', num_qubits=self.num_qubits, params=[])
        definition = QuantumCircuit(*self.definition.qregs)
        for inst in reversed(self._definition):
            definition._append(inst.replace(operation=inst.operation.inverse()))
        inverse_gate.definition = definition
        return inverse_gate

    def _get_diagonal(self):
        if False:
            for i in range(10):
                print('nop')
        (_, diag) = self._dec_mcg_up_diag()
        return diag

    def _dec_mcg_up_diag(self):
        if False:
            i = 10
            return i + 15
        '\n        Call to create a circuit with gates that implement the MCG up to a diagonal gate.\n        Remark: The qubits the gate acts on are ordered in the following way:\n            q=[q_target,q_controls,q_ancilla_zero,q_ancilla_dirty]\n        '
        diag = np.ones(2 ** (self.num_controls + 1)).tolist()
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        (q_target, q_controls, q_ancillas_zero, q_ancillas_dirty) = self._define_qubit_role(q)
        threshold = float('inf')
        if self.num_controls < threshold:
            gate_list = [np.eye(2, 2) for i in range(2 ** self.num_controls)]
            gate_list[-1] = self.params[0]
            ucg = UCGate(gate_list, up_to_diagonal=True)
            circuit.append(ucg, [q_target] + q_controls)
            diag = ucg._get_diagonal()
        return (circuit, diag)

    def _define_qubit_role(self, q):
        if False:
            return 10
        q_target = q[0]
        q_controls = q[1:self.num_controls + 1]
        q_ancillas_zero = q[self.num_controls + 1:self.num_controls + 1 + self.num_ancillas_zero]
        q_ancillas_dirty = q[self.num_controls + 1 + self.num_ancillas_zero:]
        return (q_target, q_controls, q_ancillas_zero, q_ancillas_dirty)

    def validate_parameter(self, parameter):
        if False:
            for i in range(10):
                print('nop')
        'Multi controlled single-qubit unitary gate parameter has to be an ndarray.'
        if isinstance(parameter, np.ndarray):
            return parameter
        else:
            raise CircuitError(f'invalid param type {type(parameter)} in gate {self.name}')