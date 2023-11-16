"""Depth pass testing"""
import unittest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Depth
from qiskit.test import QiskitTestCase

class TestDepthPass(QiskitTestCase):
    """Tests for Depth analysis methods."""

    def test_empty_dag(self):
        if False:
            for i in range(10):
                print('nop')
        'Empty DAG has 0 depth'
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        pass_ = Depth()
        _ = pass_.run(dag)
        self.assertEqual(pass_.property_set['depth'], 0)

    def test_just_qubits(self):
        if False:
            while True:
                i = 10
        'A dag with 8 operations and no classic bits'
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = Depth()
        _ = pass_.run(dag)
        self.assertEqual(pass_.property_set['depth'], 7)

    def test_depth_one(self):
        if False:
            return 10
        'A dag with operations in parallel and depth 1'
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = Depth()
        _ = pass_.run(dag)
        self.assertEqual(pass_.property_set['depth'], 1)

    def test_depth_control_flow(self):
        if False:
            return 10
        'A DAG with control flow still gives an estimate.'
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
            qc.cx(2, 3)
        with else_:
            qc.x(1)
            with qc.for_loop(range(3)):
                qc.z(2)
                with qc.for_loop((4, 0, 1)):
                    qc.z(2)
        with qc.while_loop((qc.clbits[0], True)):
            qc.h(0)
            qc.measure(0, 0)
        pass_ = Depth(recurse=True)
        pass_(qc)
        self.assertEqual(pass_.property_set['depth'], 16)
if __name__ == '__main__':
    unittest.main()