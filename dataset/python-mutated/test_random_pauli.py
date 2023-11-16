"""Test library of quantum volume circuits."""
import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliTwoDesign

class TestPauliTwoDesign(QiskitTestCase):
    """Test the Random Pauli circuit."""

    def test_random_pauli(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the Random Pauli circuit.'
        circuit = PauliTwoDesign(4, seed=12, reps=1)
        qr = QuantumRegister(4, 'q')
        params = circuit.ordered_parameters
        expected = QuantumCircuit(qr)
        expected.ry(np.pi / 4, qr)
        expected.ry(params[0], 0)
        expected.rx(params[1], 1)
        expected.rz(params[2], 2)
        expected.rz(params[3], 3)
        expected.cz(0, 1)
        expected.cz(2, 3)
        expected.cz(1, 2)
        expected.rx(params[4], 0)
        expected.rx(params[5], 1)
        expected.rx(params[6], 2)
        expected.rx(params[7], 3)
        self.assertEqual(circuit.decompose(), expected)

    def test_resize(self):
        if False:
            i = 10
            return i + 15
        'Test resizing the Random Pauli circuit preserves the gates.'
        circuit = PauliTwoDesign(1)
        top_gates = [instruction.operation.name for instruction in circuit.decompose().data]
        circuit.num_qubits = 3
        decomposed = circuit.decompose()
        with self.subTest('assert existing gates remain'):
            new_top_gates = []
            for instruction in decomposed:
                if instruction.qubits == (decomposed.qubits[0],):
                    new_top_gates.append(instruction.operation.name)
            self.assertEqual(top_gates, new_top_gates)

    def test_assign_keeps_one_initial_layer(self):
        if False:
            i = 10
            return i + 15
        'Test assigning parameters does not add an additional initial layer.'
        circuit = PauliTwoDesign(2)
        values = list(range(circuit.num_parameters))
        bound0 = circuit.assign_parameters(values)
        bound1 = circuit.assign_parameters(values)
        bound2 = circuit.assign_parameters(values)
        self.assertEqual(bound0, bound1)
        self.assertEqual(bound0, bound2)
if __name__ == '__main__':
    unittest.main()