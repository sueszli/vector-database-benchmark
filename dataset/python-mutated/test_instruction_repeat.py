"""Test Qiskit's repeat instruction operation."""
import unittest
from numpy import pi
from qiskit.transpiler import PassManager
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import SGate, U3Gate, CXGate, UnitaryGate
from qiskit.circuit import Instruction, Measure, Gate
from qiskit.transpiler.passes import Unroller
from qiskit.circuit.exceptions import CircuitError

class TestRepeatInt1Q(QiskitTestCase):
    """Test gate_q1.repeat() with integer"""

    def test_standard_1Q_two(self):
        if False:
            i = 10
            return i + 15
        'Test standard gate.repeat(2) method.'
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), [qr[0]])
        expected_circ.append(SGate(), [qr[0]])
        expected = expected_circ.to_instruction()
        result = SGate().repeat(2)
        self.assertEqual(result.name, 's*2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)

    def test_standard_1Q_one(self):
        if False:
            return 10
        'Test standard gate.repeat(1) method.'
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), [qr[0]])
        expected = expected_circ.to_instruction()
        result = SGate().repeat(1)
        self.assertEqual(result.name, 's*1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)

class TestRepeatInt2Q(QiskitTestCase):
    """Test gate_q2.repeat() with integer"""

    def test_standard_2Q_two(self):
        if False:
            i = 10
            return i + 15
        'Test standard 2Q gate.repeat(2) method.'
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CXGate(), [qr[0], qr[1]])
        expected_circ.append(CXGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()
        result = CXGate().repeat(2)
        self.assertEqual(result.name, 'cx*2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)

    def test_standard_2Q_one(self):
        if False:
            for i in range(10):
                print('nop')
        'Test standard 2Q gate.repeat(1) method.'
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CXGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()
        result = CXGate().repeat(1)
        self.assertEqual(result.name, 'cx*1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)

class TestRepeatIntMeasure(QiskitTestCase):
    """Test Measure.repeat() with integer"""

    def test_measure_two(self):
        if False:
            while True:
                i = 10
        'Test Measure.repeat(2) method.'
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected = expected_circ.to_instruction()
        result = Measure().repeat(2)
        self.assertEqual(result.name, 'measure*2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)
        self.assertNotIsInstance(result, Gate)

    def test_measure_one(self):
        if False:
            print('Hello World!')
        'Test Measure.repeat(1) method.'
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected = expected_circ.to_instruction()
        result = Measure().repeat(1)
        self.assertEqual(result.name, 'measure*1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)
        self.assertNotIsInstance(result, Gate)

class TestRepeatUnroller(QiskitTestCase):
    """Test unrolling Gate.repeat"""

    def test_unroller_two(self):
        if False:
            i = 10
            return i + 15
        'Test unrolling gate.repeat(2).'
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.append(SGate().repeat(2), [qr[0]])
        with self.assertWarns(DeprecationWarning):
            result = PassManager(Unroller('u3')).run(circuit)
        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, pi / 2), [qr[0]])
        expected.append(U3Gate(0, 0, pi / 2), [qr[0]])
        self.assertEqual(result, expected)

    def test_unroller_one(self):
        if False:
            i = 10
            return i + 15
        'Test unrolling gate.repeat(1).'
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.append(SGate().repeat(1), [qr[0]])
        with self.assertWarns(DeprecationWarning):
            result = PassManager(Unroller('u3')).run(circuit)
        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, pi / 2), [qr[0]])
        self.assertEqual(result, expected)

class TestRepeatErrors(QiskitTestCase):
    """Test when Gate.repeat() should raise."""

    def test_unitary_no_int(self):
        if False:
            print('Hello World!')
        'Test UnitaryGate.repeat(2/3) method. Raises, since n is not int.'
        with self.assertRaises(CircuitError) as context:
            _ = UnitaryGate([[0, 1j], [-1j, 0]]).repeat(2 / 3)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_standard_no_int(self):
        if False:
            for i in range(10):
                print('nop')
        'Test standard Gate.repeat(2/3) method. Raises, since n is not int.'
        with self.assertRaises(CircuitError) as context:
            _ = SGate().repeat(2 / 3)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_measure_zero(self):
        if False:
            while True:
                i = 10
        'Test Measure.repeat(0) method. Raises, since n<1'
        with self.assertRaises(CircuitError) as context:
            _ = Measure().repeat(0)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_standard_1Q_zero(self):
        if False:
            while True:
                i = 10
        'Test standard 2Q gate.repeat(0) method. Raises, since n<1.'
        with self.assertRaises(CircuitError) as context:
            _ = SGate().repeat(0)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_standard_1Q_minus_one(self):
        if False:
            for i in range(10):
                print('nop')
        'Test standard 2Q gate.repeat(-1) method. Raises, since n<1.'
        with self.assertRaises(CircuitError) as context:
            _ = SGate().repeat(-1)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_standard_2Q_minus_one(self):
        if False:
            return 10
        'Test standard 2Q gate.repeat(-1) method. Raises, since n<1.'
        with self.assertRaises(CircuitError) as context:
            _ = CXGate().repeat(-1)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_measure_minus_one(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Measure.repeat(-1) method. Raises, since n<1'
        with self.assertRaises(CircuitError) as context:
            _ = Measure().repeat(-1)
        self.assertIn('strictly positive integer', str(context.exception))

    def test_standard_2Q_zero(self):
        if False:
            i = 10
            return i + 15
        'Test standard 2Q gate.repeat(0) method. Raises, since n<1.'
        with self.assertRaises(CircuitError) as context:
            _ = CXGate().repeat(0)
        self.assertIn('strictly positive integer', str(context.exception))
if __name__ == '__main__':
    unittest.main()