"""Tests for quantum channel representation class."""
import numpy as np
from qiskit.quantum_info.operators.channel import SuperOp
from ..test_operator import OperatorTestCase

class ChannelTestCase(OperatorTestCase):
    """Tests for Channel representations."""
    sopI = np.eye(4)
    sopX = np.kron(OperatorTestCase.UX.conj(), OperatorTestCase.UX)
    sopY = np.kron(OperatorTestCase.UY.conj(), OperatorTestCase.UY)
    sopZ = np.kron(OperatorTestCase.UZ.conj(), OperatorTestCase.UZ)
    sopH = np.kron(OperatorTestCase.UH.conj(), OperatorTestCase.UH)
    choiI = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    choiX = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    choiY = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
    choiZ = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
    choiH = np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 2
    chiI = np.diag([2, 0, 0, 0])
    chiX = np.diag([0, 2, 0, 0])
    chiY = np.diag([0, 0, 2, 0])
    chiZ = np.diag([0, 0, 0, 2])
    chiH = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])
    ptmI = np.diag([1, 1, 1, 1])
    ptmX = np.diag([1, 1, -1, -1])
    ptmY = np.diag([1, -1, 1, -1])
    ptmZ = np.diag([1, -1, -1, 1])
    ptmH = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])

    def simple_circuit_no_measure(self):
        if False:
            i = 10
            return i + 15
        'Return a unitary circuit and the corresponding unitary array.'
        (circ, target) = super().simple_circuit_no_measure()
        return (circ, SuperOp(target))

    def depol_kraus(self, p):
        if False:
            print('Hello World!')
        'Depolarizing channel Kraus operators'
        return [np.sqrt(1 - p * 3 / 4) * self.UI, np.sqrt(p / 4) * self.UX, np.sqrt(p / 4) * self.UY, np.sqrt(p / 4) * self.UZ]

    def depol_sop(self, p):
        if False:
            return 10
        'Depolarizing channel superoperator matrix'
        return (1 - p) * self.sopI + p * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2

    def depol_choi(self, p):
        if False:
            print('Hello World!')
        'Depolarizing channel Choi-matrix'
        return (1 - p) * self.choiI + p * np.eye(4) / 2

    def depol_chi(self, p):
        if False:
            for i in range(10):
                print('nop')
        'Depolarizing channel Chi-matrix'
        return 2 * np.diag([1 - 3 * p / 4, p / 4, p / 4, p / 4])

    def depol_ptm(self, p):
        if False:
            for i in range(10):
                print('nop')
        'Depolarizing channel PTM'
        return np.diag([1, 1 - p, 1 - p, 1 - p])

    def depol_stine(self, p):
        if False:
            for i in range(10):
                print('nop')
        'Depolarizing channel Stinespring matrix'
        kraus = self.depol_kraus(p)
        basis = np.eye(4).reshape((4, 4, 1))
        return np.sum([np.kron(k, b) for (k, b) in zip(kraus, basis)], axis=0)

    def rand_kraus(self, input_dim, output_dim, n):
        if False:
            print('Hello World!')
        'Return a random (non-CPTP) Kraus operator map'
        return [self.rand_matrix(output_dim, input_dim) for _ in range(n)]