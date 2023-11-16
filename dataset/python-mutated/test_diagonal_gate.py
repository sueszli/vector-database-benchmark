"""Diagonal gate tests."""
import unittest
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute, assemble
from qiskit import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.extensions.quantum_initializer import DiagonalGate
from qiskit.quantum_info.operators.predicates import matrix_equal

class TestDiagonalGate(QiskitTestCase):
    """
    Diagonal gate tests.
    """

    def test_diag_gate(self):
        if False:
            print('Hello World!')
        'Test diagonal gates.'
        for phases in [[0, 0], [0, 0.8], [0, 0, 1, 1], [0, 1, 0.5, 1], (2 * np.pi * np.random.rand(2 ** 3)).tolist(), (2 * np.pi * np.random.rand(2 ** 4)).tolist(), (2 * np.pi * np.random.rand(2 ** 5)).tolist()]:
            with self.subTest(phases=phases):
                diag = [np.exp(1j * ph) for ph in phases]
                num_qubits = int(np.log2(len(diag)))
                q = QuantumRegister(num_qubits)
                qc = QuantumCircuit(q)
                with self.assertWarns(PendingDeprecationWarning):
                    qc.diagonal(diag, q[0:num_qubits])
                qc = transpile(qc, basis_gates=['u1', 'u3', 'u2', 'cx', 'id'], optimization_level=0)
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                unitary_desired = _get_diag_gate_matrix(diag)
                self.assertTrue(matrix_equal(unitary, unitary_desired, ignore_phase=False))

    def test_mod1_entries(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that diagonal raises if entries do not have modules of 1.'
        from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT
        with self.assertRaises(QiskitError):
            with self.assertWarns(PendingDeprecationWarning):
                DiagonalGate([1, 1 - 2 * ATOL_DEFAULT - RTOL_DEFAULT])

    def test_npcomplex_params_conversion(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify diagonal gate converts numpy.complex to complex.'
        diag = np.array([1 + 0j, 1 + 0j])
        qc = QuantumCircuit(1)
        with self.assertWarns(PendingDeprecationWarning):
            qc.diagonal(diag.tolist(), [0])
        params = qc.data[0].operation.params
        self.assertTrue(all((isinstance(p, complex) and (not isinstance(p, np.number)) for p in params)))
        qobj = assemble(qc)
        params = qobj.experiments[0].instructions[0].params
        self.assertTrue(all((isinstance(p, complex) and (not isinstance(p, np.number)) for p in params)))

def _get_diag_gate_matrix(diag):
    if False:
        while True:
            i = 10
    return np.diagflat(diag)
if __name__ == '__main__':
    unittest.main()