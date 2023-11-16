"""ResourceEstimation pass testing"""
import unittest
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ResourceEstimation
from qiskit.test import QiskitTestCase

class TestResourceEstimationPass(QiskitTestCase):
    """Tests for PropertySet methods."""

    def test_empty_dag(self):
        if False:
            i = 10
            return i + 15
        'Empty DAG.'
        circuit = QuantumCircuit()
        passmanager = PassManager()
        passmanager.append(ResourceEstimation())
        passmanager.run(circuit)
        self.assertEqual(passmanager.property_set['size'], 0)
        self.assertEqual(passmanager.property_set['depth'], 0)
        self.assertEqual(passmanager.property_set['width'], 0)
        self.assertDictEqual(passmanager.property_set['count_ops'], {})

    def test_just_qubits(self):
        if False:
            i = 10
            return i + 15
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
        passmanager = PassManager()
        passmanager.append(ResourceEstimation())
        passmanager.run(circuit)
        self.assertEqual(passmanager.property_set['size'], 8)
        self.assertEqual(passmanager.property_set['depth'], 7)
        self.assertEqual(passmanager.property_set['width'], 2)
        self.assertDictEqual(passmanager.property_set['count_ops'], {'cx': 6, 'h': 2})
if __name__ == '__main__':
    unittest.main()