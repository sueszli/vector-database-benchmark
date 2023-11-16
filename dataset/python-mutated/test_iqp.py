"""Test library of IQP circuits."""
import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import IQP
from qiskit.quantum_info import Operator

class TestIQPLibrary(QiskitTestCase):
    """Test library of IQP quantum circuits."""

    def test_iqp(self):
        if False:
            i = 10
            return i + 15
        'Test iqp circuit.'
        circuit = IQP(interactions=np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]]))
        expected = QuantumCircuit(3)
        expected.h([0, 1, 2])
        expected.cp(5 * np.pi / 2, 0, 1)
        expected.cp(3 * np.pi / 2, 1, 2)
        expected.cp(1 * np.pi / 2, 0, 2)
        expected.p(6 * np.pi / 8, 0)
        expected.p(4 * np.pi / 8, 1)
        expected.p(2 * np.pi / 8, 2)
        expected.h([0, 1, 2])
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    def test_iqp_bad(self):
        if False:
            while True:
                i = 10
        'Test that [0,..,n-1] permutation is required (no -1 for last element).'
        self.assertRaises(CircuitError, IQP, [[6, 5], [2, 4]])
if __name__ == '__main__':
    unittest.main()