"""Test the Layout Score pass"""
import unittest
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.transpiler.passes import Layout2qDistance
from qiskit.transpiler import CouplingMap, Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.target import Target
from qiskit.test import QiskitTestCase

class TestLayoutScoreError(QiskitTestCase):
    """Test error-ish of Layout Score"""

    def test_no_layout(self):
        if False:
            print('Hello World!')
        'No Layout. Empty Circuit CouplingMap map: None. Result: None'
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        coupling = CouplingMap()
        layout = None
        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set['layout'] = layout
        pass_.run(dag)
        self.assertIsNone(pass_.property_set['layout_score'])

class TestTrivialLayoutScore(QiskitTestCase):
    """Trivial layout scenarios"""

    def test_no_cx(self):
        if False:
            while True:
                i = 10
        'Empty Circuit CouplingMap map: None. Result: 0'
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        coupling = CouplingMap()
        layout = Layout().generate_trivial_layout(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set['layout'] = layout
        pass_.run(dag)
        self.assertEqual(pass_.property_set['layout_score'], 0)

    def test_swap_mapped_true(self):
        if False:
            for i in range(10):
                print('nop')
        'Mapped circuit. Good Layout\n        qr0 (0):--(+)---(+)-\n                   |     |\n        qr1 (1):---.-----|--\n                         |\n        qr2 (2):---------.--\n\n        CouplingMap map: [1]--[0]--[2]\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        layout = Layout().generate_trivial_layout(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set['layout'] = layout
        pass_.run(dag)
        self.assertEqual(pass_.property_set['layout_score'], 0)

    def test_swap_mapped_false(self):
        if False:
            while True:
                i = 10
        'Needs [0]-[1] in a [0]--[2]--[1] Result:1\n        qr0:--(+)--\n               |\n        qr1:---.---\n\n        CouplingMap map: [0]--[2]--[1]\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 2], [2, 1]])
        layout = Layout().generate_trivial_layout(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set['layout'] = layout
        pass_.run(dag)
        self.assertEqual(pass_.property_set['layout_score'], 1)

    def test_swap_mapped_true_target(self):
        if False:
            for i in range(10):
                print('nop')
        'Mapped circuit. Good Layout\n        qr0 (0):--(+)---(+)-\n                   |     |\n        qr1 (1):---.-----|--\n                         |\n        qr2 (2):---------.--\n\n        CouplingMap map: [1]--[0]--[2]\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (0, 2): None})
        layout = Layout().generate_trivial_layout(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(target)
        pass_.property_set['layout'] = layout
        pass_.run(dag)
        self.assertEqual(pass_.property_set['layout_score'], 0)

    def test_swap_mapped_false_target(self):
        if False:
            for i in range(10):
                print('nop')
        'Needs [0]-[1] in a [0]--[2]--[1] Result:1\n        qr0:--(+)--\n               |\n        qr1:---.---\n\n        CouplingMap map: [0]--[2]--[1]\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        target = Target()
        target.add_instruction(CXGate(), {(0, 2): None, (2, 1): None})
        layout = Layout().generate_trivial_layout(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(target)
        pass_.property_set['layout'] = layout
        pass_.run(dag)
        self.assertEqual(pass_.property_set['layout_score'], 1)
if __name__ == '__main__':
    unittest.main()