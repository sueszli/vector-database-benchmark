"""Tests for the converters."""
import os
import unittest
from qiskit.converters import ast_to_dag, circuit_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import qasm
from qiskit.test import QiskitTestCase

class TestAstToDag(QiskitTestCase):
    """Test AST to DAG."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.ccx(qr[0], qr[1], qr[2])
        self.circuit.measure(qr, cr)
        self.dag = circuit_to_dag(self.circuit)

    def test_from_ast_to_dag(self):
        if False:
            i = 10
            return i + 15
        'Test Unroller.execute()'
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qasm')
        ast = qasm.Qasm(os.path.join(qasm_dir, 'example.qasm')).parse()
        dag_circuit = ast_to_dag(ast)
        expected_result = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nqreg r[3];\ncreg c[3];\ncreg d[3];\nh q[0];\nh q[1];\nh q[2];\ncx q[0],r[0];\ncx q[1],r[1];\ncx q[2],r[2];\nbarrier q[0],q[1],q[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure r[0] -> d[0];\nmeasure r[1] -> d[1];\nmeasure r[2] -> d[2];\n'
        expected_dag = circuit_to_dag(QuantumCircuit.from_qasm_str(expected_result))
        self.assertEqual(dag_circuit, expected_dag)
if __name__ == '__main__':
    unittest.main(verbosity=2)