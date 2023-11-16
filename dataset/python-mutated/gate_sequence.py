"""Algebra utilities and the ``GateSequence`` class."""
from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

class GateSequence:
    """A class implementing a sequence of gates.

    This class stores the sequence of gates along with the unitary they implement.
    """

    def __init__(self, gates: Sequence[Gate]=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create a new sequence of gates.\n\n        Args:\n            gates: The gates in the sequence. The default is [].\n        '
        self.gates = list(gates)
        self.matrices = [np.asarray(gate, dtype=np.complex128) for gate in gates]
        self.labels = [gate.name for gate in gates]
        u2_matrix = np.identity(2)
        for matrix in self.matrices:
            u2_matrix = matrix.dot(u2_matrix)
        (su2_matrix, global_phase) = _convert_u2_to_su2(u2_matrix)
        so3_matrix = _convert_su2_to_so3(su2_matrix)
        self._eulers = None
        self.name = ' '.join(self.labels)
        self.global_phase = global_phase
        self.product = so3_matrix
        self.product_su2 = su2_matrix

    def remove_cancelling_pair(self, indices: Sequence[int]) -> None:
        if False:
            return 10
        'Remove a pair of indices that cancel each other and *do not* change the matrices.'
        for index in list(indices[::-1]):
            self.gates.pop(index)
            self.labels.pop(index)
        self.name = ' '.join(self.labels)

    def __eq__(self, other: 'GateSequence') -> bool:
        if False:
            return 10
        'Check if this GateSequence is the same as the other GateSequence.\n\n        Args:\n            other: The GateSequence that will be compared to ``self``.\n\n        Returns:\n            True if ``other`` is equivalent to ``self``, false otherwise.\n\n        '
        if not len(self.gates) == len(other.gates):
            return False
        for (gate1, gate2) in zip(self.gates, other.gates):
            if gate1 != gate2:
                return False
        if self.global_phase != other.global_phase:
            return False
        return True

    def to_circuit(self):
        if False:
            return 10
        'Convert to a circuit.\n\n        If no gates set but the product is not the identity, returns a circuit with a\n        unitary operation to implement the matrix.\n        '
        if len(self.gates) == 0 and (not np.allclose(self.product, np.identity(3))):
            circuit = QuantumCircuit(1, global_phase=self.global_phase)
            su2 = _convert_so3_to_su2(self.product)
            circuit.unitary(su2, [0])
            return circuit
        circuit = QuantumCircuit(1, global_phase=self.global_phase)
        for gate in self.gates:
            circuit.append(gate, [0])
        return circuit

    def to_dag(self):
        if False:
            print('Hello World!')
        'Convert to a :class:`.DAGCircuit`.\n\n        If no gates set but the product is not the identity, returns a circuit with a\n        unitary operation to implement the matrix.\n        '
        from qiskit.dagcircuit import DAGCircuit
        qreg = (Qubit(),)
        dag = DAGCircuit()
        dag.add_qubits(qreg)
        if len(self.gates) == 0 and (not np.allclose(self.product, np.identity(3))):
            su2 = _convert_so3_to_su2(self.product)
            dag.apply_operation_back(UnitaryGate(su2), qreg, check=False)
            return dag
        dag.global_phase = self.global_phase
        for gate in self.gates:
            dag.apply_operation_back(gate, qreg, check=False)
        return dag

    def append(self, gate: Gate) -> 'GateSequence':
        if False:
            return 10
        'Append gate to the sequence of gates.\n\n        Args:\n            gate: The gate to be appended.\n\n        Returns:\n            GateSequence with ``gate`` appended.\n        '
        self._eulers = None
        matrix = np.array(gate, dtype=np.complex128)
        (su2, phase) = _convert_u2_to_su2(matrix)
        so3 = _convert_su2_to_so3(su2)
        self.product = so3.dot(self.product)
        self.product_su2 = su2.dot(self.product_su2)
        self.global_phase = self.global_phase + phase
        self.gates.append(gate)
        if len(self.labels) > 0:
            self.name += f' {gate.name}'
        else:
            self.name = gate.name
        self.labels.append(gate.name)
        self.matrices.append(matrix)
        return self

    def adjoint(self) -> 'GateSequence':
        if False:
            i = 10
            return i + 15
        'Get the complex conjugate.'
        adjoint = GateSequence()
        adjoint.gates = [gate.inverse() for gate in reversed(self.gates)]
        adjoint.labels = [inv.name for inv in adjoint.gates]
        adjoint.name = ' '.join(adjoint.labels)
        adjoint.product = np.conj(self.product).T
        adjoint.product_su2 = np.conj(self.product_su2).T
        adjoint.global_phase = -self.global_phase
        return adjoint

    def copy(self) -> 'GateSequence':
        if False:
            for i in range(10):
                print('nop')
        'Create copy of the sequence of gates.\n\n        Returns:\n            A new ``GateSequence`` containing copy of list of gates.\n\n        '
        out = type(self).__new__(type(self))
        out.labels = self.labels.copy()
        out.gates = self.gates.copy()
        out.matrices = self.matrices.copy()
        out.global_phase = self.global_phase
        out.product = self.product.copy()
        out.product_su2 = self.product_su2.copy()
        out.name = self.name
        out._eulers = self._eulers
        return out

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        'Return length of sequence of gates.\n\n        Returns:\n            Length of list containing gates.\n        '
        return len(self.gates)

    def __getitem__(self, index: int) -> Gate:
        if False:
            print('Hello World!')
        'Returns the gate at ``index`` from the list of gates.\n\n        Args\n            index: Index of gate in list that will be returned.\n\n        Returns:\n            The gate at ``index`` in the list of gates.\n        '
        return self.gates[index]

    def __repr__(self) -> str:
        if False:
            return 10
        'Return string representation of this object.\n\n        Returns:\n            Representation of this sequence of gates.\n        '
        out = '['
        for gate in self.gates:
            out += gate.name
            out += ', '
        out += ']'
        out += ', product: '
        out += str(self.product)
        return out

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        'Return string representation of this object.\n\n        Returns:\n            Representation of this sequence of gates.\n        '
        out = '['
        for gate in self.gates:
            out += gate.name
            out += ', '
        out += ']'
        out += ', product: \n'
        out += str(self.product)
        return out

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'GateSequence':
        if False:
            for i in range(10):
                print('nop')
        'Initialize the gate sequence from a matrix, without a gate sequence.\n\n        Args:\n            matrix: The matrix, can be SU(2) or SO(3).\n\n        Returns:\n            A ``GateSequence`` initialized from the input matrix.\n\n        Raises:\n            ValueError: If the matrix has an invalid shape.\n        '
        instance = cls()
        if matrix.shape == (2, 2):
            instance.product = _convert_su2_to_so3(matrix)
        elif matrix.shape == (3, 3):
            instance.product = matrix
        else:
            raise ValueError(f'Matrix must have shape (3, 3) or (2, 2) but has {matrix.shape}.')
        instance.gates = []
        return instance

    def dot(self, other: 'GateSequence') -> 'GateSequence':
        if False:
            print('Hello World!')
        'Compute the dot-product with another gate sequence.\n\n        Args:\n            other: The other gate sequence.\n\n        Returns:\n            The dot-product as gate sequence.\n        '
        composed = GateSequence()
        composed.gates = other.gates + self.gates
        composed.labels = other.labels + self.labels
        composed.name = ' '.join(composed.labels)
        composed.product = np.dot(self.product, other.product)
        composed.global_phase = self.global_phase + other.global_phase
        return composed

def _convert_u2_to_su2(u2_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    if False:
        print('Hello World!')
    'Convert a U(2) matrix to SU(2) by adding a global phase.'
    z = 1 / np.sqrt(np.linalg.det(u2_matrix))
    su2_matrix = z * u2_matrix
    phase = np.arctan2(np.imag(z), np.real(z))
    return (su2_matrix, phase)

def _compute_euler_angles_from_so3(matrix: np.ndarray) -> tuple[float, float, float]:
    if False:
        return 10
    'Computes the Euler angles from the SO(3)-matrix u.\n\n    Uses the algorithm from Gregory Slabaugh,\n    see `here <https://www.gregslabaugh.net/publications/euler.pdf>`_.\n\n    Args:\n        matrix: The SO(3)-matrix for which the Euler angles need to be computed.\n\n    Returns:\n        Tuple (phi, theta, psi), where phi is rotation about z-axis, theta rotation about y-axis\n        and psi rotation about x-axis.\n    '
    matrix = np.round(matrix, decimals=10)
    if matrix[2][0] != 1 and matrix[2][1] != -1:
        theta = -math.asin(matrix[2][0])
        psi = math.atan2(matrix[2][1] / math.cos(theta), matrix[2][2] / math.cos(theta))
        phi = math.atan2(matrix[1][0] / math.cos(theta), matrix[0][0] / math.cos(theta))
        return (phi, theta, psi)
    else:
        phi = 0
        if matrix[2][0] == 1:
            theta = math.pi / 2
            psi = phi + math.atan2(matrix[0][1], matrix[0][2])
        else:
            theta = -math.pi / 2
            psi = -phi + math.atan2(-matrix[0][1], -matrix[0][2])
        return (phi, theta, psi)

def _compute_su2_from_euler_angles(angles: tuple[float, float, float]) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Computes SU(2)-matrix from Euler angles.\n\n    Args:\n        angles: The tuple containing the Euler angles for which the corresponding SU(2)-matrix\n            needs to be computed.\n\n    Returns:\n        The SU(2)-matrix corresponding to the Euler angles in angles.\n    '
    (phi, theta, psi) = angles
    uz_phi = np.array([[np.exp(-0.5j * phi), 0], [0, np.exp(0.5j * phi)]], dtype=complex)
    uy_theta = np.array([[math.cos(theta / 2), math.sin(theta / 2)], [-math.sin(theta / 2), math.cos(theta / 2)]], dtype=complex)
    ux_psi = np.array([[math.cos(psi / 2), math.sin(psi / 2) * 1j], [math.sin(psi / 2) * 1j, math.cos(psi / 2)]], dtype=complex)
    return np.dot(uz_phi, np.dot(uy_theta, ux_psi))

def _convert_su2_to_so3(matrix: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    'Computes SO(3)-matrix from input SU(2)-matrix.\n\n    Args:\n        matrix: The SU(2)-matrix for which a corresponding SO(3)-matrix needs to be computed.\n\n    Returns:\n        The SO(3)-matrix corresponding to ``matrix``.\n\n    Raises:\n        ValueError: if ``matrix`` is not an SU(2)-matrix.\n    '
    _check_is_su2(matrix)
    matrix = matrix.astype(complex)
    a = np.real(matrix[0][0])
    b = np.imag(matrix[0][0])
    c = -np.real(matrix[0][1])
    d = -np.imag(matrix[0][1])
    rotation = np.array([[a ** 2 - b ** 2 - c ** 2 + d ** 2, 2 * a * b + 2 * c * d, -2 * a * c + 2 * b * d], [-2 * a * b + 2 * c * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * a * d + 2 * b * c], [2 * a * c + 2 * b * d, 2 * b * c - 2 * a * d, a ** 2 + b ** 2 - c ** 2 - d ** 2]], dtype=float)
    return rotation

def _convert_so3_to_su2(matrix: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Converts an SO(3)-matrix to a corresponding SU(2)-matrix.\n\n    Args:\n        matrix: SO(3)-matrix to convert.\n\n    Returns:\n        SU(2)-matrix corresponding to SO(3)-matrix ``matrix``.\n\n    Raises:\n        ValueError: if ``matrix`` is not an SO(3)-matrix.\n    '
    _check_is_so3(matrix)
    return _compute_su2_from_euler_angles(_compute_euler_angles_from_so3(matrix))

def _check_is_su2(matrix: np.ndarray) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Check whether ``matrix`` is SU(2), otherwise raise an error.'
    if matrix.shape != (2, 2):
        raise ValueError(f'Matrix must have shape (2, 2) but has {matrix.shape}.')
    if abs(np.linalg.det(matrix) - 1) > 0.0001:
        raise ValueError(f'Determinant of matrix must be 1, but is {np.linalg.det(matrix)}.')

def _check_is_so3(matrix: np.ndarray) -> None:
    if False:
        while True:
            i = 10
    'Check whether ``matrix`` is SO(3), otherwise raise an error.'
    if matrix.shape != (3, 3):
        raise ValueError(f'Matrix must have shape (3, 3) but has {matrix.shape}.')
    if abs(np.linalg.det(matrix) - 1) > 0.0001:
        raise ValueError(f'Determinant of matrix must be 1, but is {np.linalg.det(matrix)}.')