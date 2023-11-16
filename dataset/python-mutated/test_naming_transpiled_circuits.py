"""Testing naming functionality of transpiled circuits"""
import unittest
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit import BasicAer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.test import QiskitTestCase

class TestNamingTranspiledCircuits(QiskitTestCase):
    """Testing the naming fuctionality for transpiled circuits."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.basis_gates = ['u1', 'u2', 'u3', 'cx']
        self.backend = BasicAer.get_backend('qasm_simulator')
        self.circuit0 = QuantumCircuit(name='circuit0')
        self.circuit1 = QuantumCircuit(name='circuit1')
        self.circuit2 = QuantumCircuit(name='circuit2')
        self.circuit3 = QuantumCircuit(name='circuit3')

    def test_single_circuit_name_singleton(self):
        if False:
            print('Hello World!')
        'Test output_name with a single circuit\n        Given a single circuit and a output name in form of a string, this test\n        checks whether that string name is assigned to the transpiled circuit.\n        '
        trans_cirkie = transpile(self.circuit0, basis_gates=self.basis_gates, output_name='transpiled-cirkie')
        self.assertEqual(trans_cirkie.name, 'transpiled-cirkie')

    def test_single_circuit_name_list(self):
        if False:
            i = 10
            return i + 15
        'Test singleton output_name and a single circuit\n        Given a single circuit and an output name in form of a single element\n        list, this test checks whether the transpiled circuit is mapped with\n        that assigned name in the list.\n        If list has more than one element, then test checks whether the\n        Transpile function raises an error.\n        '
        trans_cirkie = transpile(self.circuit0, basis_gates=self.basis_gates, output_name=['transpiled-cirkie'])
        self.assertEqual(trans_cirkie.name, 'transpiled-cirkie')

    def test_single_circuit_and_multiple_name_list(self):
        if False:
            return 10
        'Test multiple output_name and a single circuit'
        with self.assertRaises(TranspilerError):
            transpile(self.circuit0, basis_gates=self.basis_gates, output_name=['cool-cirkie', 'new-cirkie', 'dope-cirkie', 'awesome-cirkie'])

    def test_multiple_circuits_name_singleton(self):
        if False:
            i = 10
            return i + 15
        'Test output_name raise error if a single name is provided to a list of circuits\n        Given multiple circuits and a single string as a name, this test checks\n        whether the Transpile function raises an error.\n        '
        with self.assertRaises(TranspilerError):
            transpile([self.circuit1, self.circuit2], self.backend, output_name='circ')

    def test_multiple_circuits_name_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Test output_name with a list of circuits\n        Given multiple circuits and a list for output names, if\n        len(list)=len(circuits), then test checks whether transpile func assigns\n        each element in list to respective circuit.\n        If lengths are not equal, then test checks whether transpile func raises\n        error.\n        '
        circuits = [self.circuit1, self.circuit2, self.circuit3]
        names = ['awesome-circ1', 'awesome-circ2', 'awesome-circ3']
        trans_circuits = transpile(circuits, self.backend, output_name=names)
        self.assertEqual(trans_circuits[0].name, 'awesome-circ1')
        self.assertEqual(trans_circuits[1].name, 'awesome-circ2')
        self.assertEqual(trans_circuits[2].name, 'awesome-circ3')

    def test_greater_circuits_name_list(self):
        if False:
            print('Hello World!')
        'Test output_names list greater than circuits list'
        circuits = [self.circuit1, self.circuit2, self.circuit3]
        names = ['awesome-circ1', 'awesome-circ2', 'awesome-circ3', 'awesome-circ4']
        with self.assertRaises(TranspilerError):
            transpile(circuits, self.backend, output_name=names)

    def test_smaller_circuits_name_list(self):
        if False:
            while True:
                i = 10
        'Test output_names list smaller than circuits list'
        circuits = [self.circuit1, self.circuit2, self.circuit3]
        names = ['awesome-circ1', 'awesome-circ2']
        with self.assertRaises(TranspilerError):
            transpile(circuits, self.backend, output_name=names)
if __name__ == '__main__':
    unittest.main()