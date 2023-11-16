"""DAGFixedPoint pass testing"""
import unittest
from qiskit.transpiler.passes import DAGFixedPoint
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

class TestFixedPointPass(QiskitTestCase):
    """Tests for PropertySet methods."""

    def test_empty_dag_true(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the dag fixed point of an empty dag.'
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        pass_ = DAGFixedPoint()
        pass_.run(dag)
        self.assertFalse(pass_.property_set['dag_fixed_point'])
        pass_.run(dag)
        self.assertTrue(pass_.property_set['dag_fixed_point'])

    def test_nonempty_dag_false(self):
        if False:
            while True:
                i = 10
        'Test the dag false fixed point of a non-empty dag.'
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = DAGFixedPoint()
        pass_.run(dag)
        self.assertFalse(pass_.property_set['dag_fixed_point'])
        dag.remove_all_ops_named('h')
        pass_.run(dag)
        self.assertFalse(pass_.property_set['dag_fixed_point'])
if __name__ == '__main__':
    unittest.main()