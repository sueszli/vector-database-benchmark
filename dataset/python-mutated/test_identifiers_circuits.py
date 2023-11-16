"""Non-string identifiers for circuit and record identifiers test"""
import unittest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase

class TestAnonymousIds(QiskitTestCase):
    """Test the anonymous use of registers."""

    def test_create_anonymous_classical_register(self):
        if False:
            while True:
                i = 10
        'ClassicalRegister with no name.'
        cr = ClassicalRegister(size=3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_anonymous_quantum_register(self):
        if False:
            i = 10
            return i + 15
        'QuantumRegister with no name.'
        qr = QuantumRegister(size=3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_create_anonymous_classical_registers(self):
        if False:
            i = 10
            return i + 15
        'Several ClassicalRegister with no name.'
        cr1 = ClassicalRegister(size=3)
        cr2 = ClassicalRegister(size=3)
        self.assertNotEqual(cr1.name, cr2.name)

    def test_create_anonymous_quantum_registers(self):
        if False:
            print('Hello World!')
        'Several QuantumRegister with no name.'
        qr1 = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)
        self.assertNotEqual(qr1.name, qr2.name)

    def test_create_anonymous_mixed_registers(self):
        if False:
            i = 10
            return i + 15
        'Several Registers with no name.'
        cr0 = ClassicalRegister(size=3)
        qr0 = QuantumRegister(size=3)
        cr_index = int(cr0.name[1:])
        qr_index = int(qr0.name[1:])
        cr1 = ClassicalRegister(size=3)
        _ = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)
        cr_current = int(cr1.name[1:])
        qr_current = int(qr2.name[1:])
        self.assertEqual(cr_current, cr_index + 1)
        self.assertEqual(qr_current, qr_index + 2)

    def test_create_circuit_noname(self):
        if False:
            return 10
        'Create_circuit with no name.'
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        qc = QuantumCircuit(qr, cr)
        self.assertIsInstance(qc, QuantumCircuit)

class TestInvalidIds(QiskitTestCase):
    """Circuits and records with invalid IDs"""

    def test_invalid_type_circuit_name(self):
        if False:
            while True:
                i = 10
        'QuantumCircuit() with invalid type name.'
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        self.assertRaises(CircuitError, QuantumCircuit, qr, cr, name=1)
if __name__ == '__main__':
    unittest.main(verbosity=2)