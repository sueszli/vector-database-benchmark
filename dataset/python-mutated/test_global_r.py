"""Test the global rotation circuit."""
import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GR, GRX, GRY, GRZ, RGate, RZGate

class TestGlobalRLibrary(QiskitTestCase):
    """Test library of global R gates."""

    def test_gr_equivalence(self):
        if False:
            for i in range(10):
                print('nop')
        'Test global R gate is same as 3 individual R gates.'
        circuit = GR(num_qubits=3, theta=np.pi / 3, phi=2 * np.pi / 3)
        expected = QuantumCircuit(3, name='gr')
        for i in range(3):
            expected.append(RGate(theta=np.pi / 3, phi=2 * np.pi / 3), [i])
        self.assertEqual(expected, circuit.decompose())

    def test_grx_equivalence(self):
        if False:
            for i in range(10):
                print('nop')
        'Test global RX gates is same as 3 individual RX gates.'
        circuit = GRX(num_qubits=3, theta=np.pi / 3)
        expected = GR(num_qubits=3, theta=np.pi / 3, phi=0)
        self.assertEqual(expected, circuit)

    def test_gry_equivalence(self):
        if False:
            while True:
                i = 10
        'Test global RY gates is same as 3 individual RY gates.'
        circuit = GRY(num_qubits=3, theta=np.pi / 3)
        expected = GR(num_qubits=3, theta=np.pi / 3, phi=np.pi / 2)
        self.assertEqual(expected, circuit)

    def test_grz_equivalence(self):
        if False:
            while True:
                i = 10
        'Test global RZ gate is same as 3 individual RZ gates.'
        circuit = GRZ(num_qubits=3, phi=2 * np.pi / 3)
        expected = QuantumCircuit(3, name='grz')
        for i in range(3):
            expected.append(RZGate(phi=2 * np.pi / 3), [i])
        self.assertEqual(expected, circuit)
if __name__ == '__main__':
    unittest.main()