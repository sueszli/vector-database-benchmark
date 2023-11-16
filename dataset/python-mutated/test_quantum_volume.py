"""Test library of quantum volume circuits."""
import unittest
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Operator
from qiskit.quantum_info.random import random_unitary

class TestQuantumVolumeLibrary(QiskitTestCase):
    """Test library of quantum volume quantum circuits."""

    def test_qv(self):
        if False:
            print('Hello World!')
        'Test qv circuit.'
        circuit = QuantumVolume(2, 2, seed=2, classical_permutation=False)
        expected = QuantumCircuit(2)
        expected.swap(0, 1)
        expected.append(random_unitary(4, seed=837), [0, 1])
        expected.append(random_unitary(4, seed=262), [0, 1])
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))
if __name__ == '__main__':
    unittest.main()