"""Test the LayoutTransformation pass"""
import unittest
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap, Layout, Target
from qiskit.circuit.library import CXGate
from qiskit.transpiler.passes import LayoutTransformation

class TestLayoutTransformation(QiskitTestCase):
    """
    Tests the LayoutTransformation pass.
    """

    def test_three_qubit(self):
        if False:
            while True:
                i = 10
        'Test if the permutation {0->2,1->0,2->1} is implemented correctly.'
        v = QuantumRegister(3, 'v')
        coupling = CouplingMap([[0, 1], [1, 2]])
        from_layout = Layout({v[0]: 0, v[1]: 1, v[2]: 2})
        to_layout = Layout({v[0]: 2, v[1]: 0, v[2]: 1})
        ltpass = LayoutTransformation(coupling_map=coupling, from_layout=from_layout, to_layout=to_layout, seed=42)
        qc = QuantumCircuit(3)
        dag = circuit_to_dag(qc)
        output_dag = ltpass.run(dag)
        expected = QuantumCircuit(3)
        expected.swap(1, 0)
        expected.swap(1, 2)
        self.assertEqual(circuit_to_dag(expected), output_dag)

    def test_four_qubit(self):
        if False:
            while True:
                i = 10
        'Test if the permutation {0->3,1->0,2->1,3->2} is implemented correctly.'
        v = QuantumRegister(4, 'v')
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        from_layout = Layout({v[0]: 0, v[1]: 1, v[2]: 2, v[3]: 3})
        to_layout = Layout({v[0]: 3, v[1]: 0, v[2]: 1, v[3]: 2})
        ltpass = LayoutTransformation(coupling_map=coupling, from_layout=from_layout, to_layout=to_layout, seed=42)
        qc = QuantumCircuit(4)
        dag = circuit_to_dag(qc)
        output_dag = ltpass.run(dag)
        expected = QuantumCircuit(4)
        expected.swap(1, 0)
        expected.swap(1, 2)
        expected.swap(2, 3)
        self.assertEqual(circuit_to_dag(expected), output_dag)

    def test_four_qubit_with_target(self):
        if False:
            return 10
        'Test if the permutation {0->3,1->0,2->1,3->2} is implemented correctly.'
        v = QuantumRegister(4, 'v')
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None, (2, 3): None})
        from_layout = Layout({v[0]: 0, v[1]: 1, v[2]: 2, v[3]: 3})
        to_layout = Layout({v[0]: 3, v[1]: 0, v[2]: 1, v[3]: 2})
        ltpass = LayoutTransformation(target, from_layout=from_layout, to_layout=to_layout, seed=42)
        qc = QuantumCircuit(4)
        dag = circuit_to_dag(qc)
        output_dag = ltpass.run(dag)
        expected = QuantumCircuit(4)
        expected.swap(1, 0)
        expected.swap(1, 2)
        expected.swap(2, 3)
        self.assertEqual(circuit_to_dag(expected), output_dag)

    @unittest.skip('rustworkx token_swapper produces correct, but sometimes random output')
    def test_full_connected_coupling_map(self):
        if False:
            print('Hello World!')
        'Test if the permutation {0->3,1->0,2->1,3->2} in a fully connected map.'
        v = QuantumRegister(4, 'v')
        from_layout = Layout({v[0]: 0, v[1]: 1, v[2]: 2, v[3]: 3})
        to_layout = Layout({v[0]: 3, v[1]: 0, v[2]: 1, v[3]: 2})
        ltpass = LayoutTransformation(coupling_map=None, from_layout=from_layout, to_layout=to_layout, seed=42)
        qc = QuantumCircuit(4)
        dag = circuit_to_dag(qc)
        output_dag = ltpass.run(dag)
        expected = QuantumCircuit(4)
        expected.swap(1, 0)
        expected.swap(2, 1)
        expected.swap(3, 2)
        self.assertEqual(circuit_to_dag(expected), output_dag)
if __name__ == '__main__':
    unittest.main()