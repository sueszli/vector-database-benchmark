"""Tests for local invariance routines."""
import unittest
from numpy.testing import assert_allclose
from qiskit.execute_function import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.test import QiskitTestCase
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.synthesis.local_invariance import two_qubit_local_invariants

class TestLocalInvariance(QiskitTestCase):
    """Test local invariance routines"""

    def test_2q_local_invariance_simple(self):
        if False:
            i = 10
            return i + 15
        'Check the local invariance parameters\n        for known simple cases.\n        '
        sim = UnitarySimulatorPy()
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [1, 0, 3])
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [0, 0, 1])
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        qc.cx(qr[0], qr[1])
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [0, 0, -1])
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.swap(qr[1], qr[0])
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [-1, 0, -3])
if __name__ == '__main__':
    unittest.main()