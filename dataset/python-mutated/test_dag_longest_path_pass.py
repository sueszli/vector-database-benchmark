"""DAGFixedPoint pass testing"""
import unittest
from qiskit.transpiler.passes import DAGLongestPath
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

class TestDAGLongestPathPass(QiskitTestCase):
    """Tests for PropertySet methods."""

    def test_empty_dag_true(self):
        if False:
            while True:
                i = 10
        'Test the dag longest path of an empty dag.'
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        pass_ = DAGLongestPath()
        pass_.run(dag)
        self.assertListEqual(pass_.property_set['dag_longest_path'], [])

    def test_nonempty_dag_false(self):
        if False:
            while True:
                i = 10
        'Test the dag longest path non-empty dag.\n        path length = 11 = 9 ops + 2 qubits at start and end of path\n        '
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
        pass_ = DAGLongestPath()
        pass_.run(dag)
        self.assertEqual(len(pass_.property_set['dag_longest_path']), 11)
if __name__ == '__main__':
    unittest.main()