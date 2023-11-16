"""Depth pass testing"""
import unittest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CountOpsLongestPath
from qiskit.test import QiskitTestCase

class TestCountOpsLongestPathPass(QiskitTestCase):
    """Tests for CountOpsLongestPath analysis methods."""

    def test_empty_dag(self):
        if False:
            print('Hello World!')
        'Empty DAG has empty counts.'
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        pass_ = CountOpsLongestPath()
        _ = pass_.run(dag)
        self.assertDictEqual(pass_.property_set['count_ops_longest_path'], {})

    def test_just_qubits(self):
        if False:
            i = 10
            return i + 15
        'A dag with 9 operations (3 CXs, 2Xs, 2Ys and 2 Hs) on the longest\n        path\n        '
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])
        circuit.y(qr[1])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = CountOpsLongestPath()
        _ = pass_.run(dag)
        count_ops = pass_.property_set['count_ops_longest_path']
        self.assertDictEqual(count_ops, {'cx': 3, 'x': 2, 'y': 2, 'h': 2})
if __name__ == '__main__':
    unittest.main()