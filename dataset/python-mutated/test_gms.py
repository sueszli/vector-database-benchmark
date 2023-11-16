"""Test library of Global Mølmer–Sørensen gate."""
import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
from qiskit.circuit.library import GMS, RXXGate
from qiskit.quantum_info import Operator

class TestGMSLibrary(QiskitTestCase):
    """Test library of Global Mølmer–Sørensen gate."""

    def test_twoq_equivalence(self):
        if False:
            print('Hello World!')
        'Test GMS on 2 qubits is same as RXX.'
        circuit = GMS(num_qubits=2, theta=[[0, np.pi / 3], [0, 0]])
        expected = RXXGate(np.pi / 3)
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))
if __name__ == '__main__':
    unittest.main()