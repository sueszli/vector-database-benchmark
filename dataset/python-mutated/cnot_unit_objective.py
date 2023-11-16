"""
A definition of the approximate circuit compilation optimization problem based on CNOT unit
definition.
"""
from __future__ import annotations
import typing
from abc import ABC
import numpy as np
from numpy import linalg as la
from .approximate import ApproximatingObjective
from .elementary_operations import ry_matrix, rz_matrix, place_unitary, place_cnot, rx_matrix

class CNOTUnitObjective(ApproximatingObjective, ABC):
    """
    A base class for a problem definition based on CNOT unit. This class may have different
    subclasses for objective and gradient computations.
    """

    def __init__(self, num_qubits: int, cnots: np.ndarray) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            num_qubits: number of qubits.\n            cnots: a CNOT structure to be used in the optimization procedure.\n        '
        super().__init__()
        self._num_qubits = num_qubits
        self._cnots = cnots
        self._num_cnots = cnots.shape[1]

    @property
    def num_cnots(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            A number of CNOT units to be used by the approximate circuit.\n        '
        return self._num_cnots

    @property
    def num_thetas(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n            Number of parameters (angles) of rotation gates in this circuit.\n        '
        return 3 * self._num_qubits + 4 * self._num_cnots

class DefaultCNOTUnitObjective(CNOTUnitObjective):
    """A naive implementation of the objective function based on CNOT units."""

    def __init__(self, num_qubits: int, cnots: np.ndarray) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            num_qubits: number of qubits.\n            cnots: a CNOT structure to be used in the optimization procedure.\n        '
        super().__init__(num_qubits, cnots)
        self._last_thetas: np.ndarray | None = None
        self._cnot_right_collection: np.ndarray | None = None
        self._cnot_left_collection: np.ndarray | None = None
        self._rotation_matrix: int | np.ndarray | None = None
        self._cnot_matrix: np.ndarray | None = None

    def objective(self, param_values: np.ndarray) -> typing.SupportsFloat:
        if False:
            while True:
                i = 10
        thetas = param_values
        n = self._num_qubits
        d = int(2 ** n)
        cnots = self._cnots
        num_cnots = self.num_cnots
        cnot_unit_collection = np.zeros((d, d * num_cnots), dtype=complex)
        cnot_right_collection = np.zeros((d, d * num_cnots), dtype=complex)
        cnot_left_collection = np.zeros((d, d * num_cnots), dtype=complex)
        for cnot_index in range(num_cnots):
            theta_index = 4 * cnot_index
            q1 = int(cnots[0, cnot_index])
            q2 = int(cnots[1, cnot_index])
            ry1 = ry_matrix(thetas[0 + theta_index])
            rz1 = rz_matrix(thetas[1 + theta_index])
            ry2 = ry_matrix(thetas[2 + theta_index])
            rx2 = rx_matrix(thetas[3 + theta_index])
            single_q1 = np.dot(rz1, ry1)
            single_q2 = np.dot(rx2, ry2)
            full_q1 = place_unitary(single_q1, n, q1)
            full_q2 = place_unitary(single_q2, n, q2)
            cnot_q1q2 = place_cnot(n, q1, q2)
            cnot_unit_collection[:, d * cnot_index:d * (cnot_index + 1)] = la.multi_dot([full_q2, full_q1, cnot_q1q2])
        cnot_matrix = np.eye(d)
        for cnot_index in range(num_cnots - 1, -1, -1):
            cnot_matrix = np.dot(cnot_matrix, cnot_unit_collection[:, d * cnot_index:d * (cnot_index + 1)])
            cnot_right_collection[:, d * cnot_index:d * (cnot_index + 1)] = cnot_matrix
        cnot_matrix = np.eye(d)
        for cnot_index in range(num_cnots):
            cnot_matrix = np.dot(cnot_unit_collection[:, d * cnot_index:d * (cnot_index + 1)], cnot_matrix)
            cnot_left_collection[:, d * cnot_index:d * (cnot_index + 1)] = cnot_matrix
        rotation_matrix: int | np.ndarray = 1
        for q in range(n):
            theta_index = 4 * num_cnots + 3 * q
            rz0 = rz_matrix(thetas[0 + theta_index])
            ry1 = ry_matrix(thetas[1 + theta_index])
            rz2 = rz_matrix(thetas[2 + theta_index])
            rotation_matrix = np.kron(rotation_matrix, la.multi_dot([rz0, ry1, rz2]))
        circuit_matrix = np.dot(cnot_matrix, rotation_matrix)
        error = 0.5 * la.norm(circuit_matrix - self._target_matrix, 'fro') ** 2
        self._last_thetas = thetas
        self._cnot_left_collection = cnot_left_collection
        self._cnot_right_collection = cnot_right_collection
        self._rotation_matrix = rotation_matrix
        self._cnot_matrix = cnot_matrix
        return error

    def gradient(self, param_values: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        thetas = param_values
        if not np.all(np.isclose(thetas, self._last_thetas)):
            self.objective(thetas)
        pauli_x = np.multiply(-1j / 2, np.asarray([[0, 1], [1, 0]]))
        pauli_y = np.multiply(-1j / 2, np.asarray([[0, -1j], [1j, 0]]))
        pauli_z = np.multiply(-1j / 2, np.asarray([[1, 0], [0, -1]]))
        n = self._num_qubits
        d = int(2 ** n)
        cnots = self._cnots
        num_cnots = self.num_cnots
        der = np.zeros(4 * num_cnots + 3 * n)
        for cnot_index in range(num_cnots):
            theta_index = 4 * cnot_index
            q1 = int(cnots[0, cnot_index])
            q2 = int(cnots[1, cnot_index])
            ry1 = ry_matrix(thetas[0 + theta_index])
            rz1 = rz_matrix(thetas[1 + theta_index])
            ry2 = ry_matrix(thetas[2 + theta_index])
            rx2 = rx_matrix(thetas[3 + theta_index])
            for i in range(4):
                if i == 0:
                    single_q1 = la.multi_dot([rz1, pauli_y, ry1])
                    single_q2 = np.dot(rx2, ry2)
                elif i == 1:
                    single_q1 = la.multi_dot([pauli_z, rz1, ry1])
                    single_q2 = np.dot(rx2, ry2)
                elif i == 2:
                    single_q1 = np.dot(rz1, ry1)
                    single_q2 = la.multi_dot([rx2, pauli_y, ry2])
                else:
                    single_q1 = np.dot(rz1, ry1)
                    single_q2 = la.multi_dot([pauli_x, rx2, ry2])
                full_q1 = place_unitary(single_q1, n, q1)
                full_q2 = place_unitary(single_q2, n, q2)
                cnot_q1q2 = place_cnot(n, q1, q2)
                der_cnot_unit = la.multi_dot([full_q2, full_q1, cnot_q1q2])
                if cnot_index == 0:
                    der_cnot_matrix = np.dot(self._cnot_right_collection[:, d:2 * d], der_cnot_unit)
                elif num_cnots - 1 == cnot_index:
                    der_cnot_matrix = np.dot(der_cnot_unit, self._cnot_left_collection[:, d * (num_cnots - 2):d * (num_cnots - 1)])
                else:
                    der_cnot_matrix = la.multi_dot([self._cnot_right_collection[:, d * (cnot_index + 1):d * (cnot_index + 2)], der_cnot_unit, self._cnot_left_collection[:, d * (cnot_index - 1):d * cnot_index]])
                der_circuit_matrix = np.dot(der_cnot_matrix, self._rotation_matrix)
                der[i + theta_index] = -np.real(np.trace(np.dot(der_circuit_matrix.conj().T, self._target_matrix)))
        for i in range(3 * n):
            der_rotation_matrix: int | np.ndarray = 1
            for q in range(n):
                theta_index = 4 * num_cnots + 3 * q
                rz0 = rz_matrix(thetas[0 + theta_index])
                ry1 = ry_matrix(thetas[1 + theta_index])
                rz2 = rz_matrix(thetas[2 + theta_index])
                if i - 3 * q == 0:
                    rz0 = np.dot(pauli_z, rz0)
                elif i - 3 * q == 1:
                    ry1 = np.dot(pauli_y, ry1)
                elif i - 3 * q == 2:
                    rz2 = np.dot(pauli_z, rz2)
                der_rotation_matrix = np.kron(der_rotation_matrix, la.multi_dot([rz0, ry1, rz2]))
            der_circuit_matrix = np.dot(self._cnot_matrix, der_rotation_matrix)
            der[4 * num_cnots + i] = -np.real(np.trace(np.dot(der_circuit_matrix.conj().T, self._target_matrix)))
        return der