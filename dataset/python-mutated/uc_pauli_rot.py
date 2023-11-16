"""Uniformly controlled Pauli rotations."""
from __future__ import annotations
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
_EPS = 1e-10

class UCPauliRotGate(Gate):
    """Uniformly controlled Pauli rotations.

    Implements the :class:`.UCGate` for the special case that all unitaries are Pauli rotations,
    :math:`U_i = R_P(a_i)` where :math:`P \\in \\{X, Y, Z\\}` and :math:`a_i \\in \\mathbb{R}` is
    the rotation angle.
    """

    def __init__(self, angle_list: list[float], rot_axis: str) -> None:
        if False:
            return 10
        '\n        Args:\n            angle_list: List of rotation angles :math:`[a_0, ..., a_{2^{k-1}}]`.\n            rot_axis: Rotation axis. Must be either of ``"X"``, ``"Y"`` or ``"Z"``.\n        '
        self.rot_axes = rot_axis
        if not isinstance(angle_list, list):
            raise QiskitError('The angles are not provided in a list.')
        for angle in angle_list:
            try:
                float(angle)
            except TypeError as ex:
                raise QiskitError('An angle cannot be converted to type float (real angles are expected).') from ex
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError('The number of controlled rotation gates is not a non-negative power of 2.')
        if rot_axis not in ('X', 'Y', 'Z'):
            raise QiskitError('Rotation axis is not supported.')
        num_qubits = int(num_contr) + 1
        super().__init__('ucr' + rot_axis.lower(), num_qubits, angle_list)

    def _define(self):
        if False:
            for i in range(10):
                print('nop')
        ucr_circuit = self._dec_ucrot()
        gate = ucr_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        ucr_circuit = QuantumCircuit(q)
        ucr_circuit.append(gate, q[:])
        self.definition = ucr_circuit

    def _dec_ucrot(self):
        if False:
            i = 10
            return i + 15
        '\n        Finds a decomposition of a UC rotation gate into elementary gates\n        (C-NOTs and single-qubit rotations).\n        '
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        q_target = q[0]
        q_controls = q[1:]
        if not q_controls:
            if self.rot_axes == 'X':
                if np.abs(self.params[0]) > _EPS:
                    circuit.rx(self.params[0], q_target)
            if self.rot_axes == 'Y':
                if np.abs(self.params[0]) > _EPS:
                    circuit.ry(self.params[0], q_target)
            if self.rot_axes == 'Z':
                if np.abs(self.params[0]) > _EPS:
                    circuit.rz(self.params[0], q_target)
        else:
            angles = self.params.copy()
            UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
            for (i, angle) in enumerate(angles):
                if self.rot_axes == 'X':
                    if np.abs(angle) > _EPS:
                        circuit.rx(angle, q_target)
                if self.rot_axes == 'Y':
                    if np.abs(angle) > _EPS:
                        circuit.ry(angle, q_target)
                if self.rot_axes == 'Z':
                    if np.abs(angle) > _EPS:
                        circuit.rz(angle, q_target)
                if not i == len(angles) - 1:
                    binary_rep = np.binary_repr(i + 1)
                    q_contr_index = len(binary_rep) - len(binary_rep.rstrip('0'))
                else:
                    q_contr_index = len(q_controls) - 1
                if self.rot_axes == 'X':
                    circuit.ry(np.pi / 2, q_target)
                circuit.cx(q_controls[q_contr_index], q_target)
                if self.rot_axes == 'X':
                    circuit.ry(-np.pi / 2, q_target)
        return circuit

    @staticmethod
    def _dec_uc_rotations(angles, start_index, end_index, reversed_dec):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at\n        the end of the circuit. The rotation angles of the gate R_t are stored in\n        angles[start_index:end_index]. If reversed_dec == True, it decomposes the gate such that\n        there is a C-NOT gate at the start of the circuit (in fact, the circuit topology for\n        the reversed decomposition is the reversed one of the original decomposition)\n        '
        interval_len_half = (end_index - start_index) // 2
        for i in range(start_index, start_index + interval_len_half):
            if not reversed_dec:
                (angles[i], angles[i + interval_len_half]) = UCPauliRotGate._update_angles(angles[i], angles[i + interval_len_half])
            else:
                (angles[i + interval_len_half], angles[i]) = UCPauliRotGate._update_angles(angles[i], angles[i + interval_len_half])
        if interval_len_half <= 1:
            return
        else:
            UCPauliRotGate._dec_uc_rotations(angles, start_index, start_index + interval_len_half, False)
            UCPauliRotGate._dec_uc_rotations(angles, start_index + interval_len_half, end_index, True)

    @staticmethod
    def _update_angles(angle1, angle2):
        if False:
            i = 10
            return i + 15
        "Calculate the new rotation angles according to Shende's decomposition."
        return ((angle1 + angle2) / 2.0, (angle1 - angle2) / 2.0)