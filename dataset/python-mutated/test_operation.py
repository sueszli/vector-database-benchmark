"""Test Qiskit's Operation class."""
import unittest
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit, Barrier, Measure, Reset, Gate, Operation
from qiskit.circuit.library import XGate, CXGate, Initialize, Isometry
from qiskit.quantum_info.operators import Clifford, CNOTDihedral, Pauli

class TestOperationClass(QiskitTestCase):
    """Testing qiskit.circuit.Operation"""

    def test_measure_as_operation(self):
        if False:
            while True:
                i = 10
        'Test that we can instantiate an object of class\n        :class:`~qiskit.circuit.Measure` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        op = Measure()
        self.assertTrue(op.name == 'measure')
        self.assertTrue(op.num_qubits == 1)
        self.assertTrue(op.num_clbits == 1)
        self.assertIsInstance(op, Operation)

    def test_reset_as_operation(self):
        if False:
            print('Hello World!')
        'Test that we can instantiate an object of class\n        :class:`~qiskit.circuit.Reset` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        op = Reset()
        self.assertTrue(op.name == 'reset')
        self.assertTrue(op.num_qubits == 1)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_barrier_as_operation(self):
        if False:
            print('Hello World!')
        'Test that we can instantiate an object of class\n        :class:`~qiskit.circuit.Barrier` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        num_qubits = 4
        op = Barrier(num_qubits)
        self.assertTrue(op.name == 'barrier')
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_clifford_as_operation(self):
        if False:
            while True:
                i = 10
        'Test that we can instantiate an object of class\n        :class:`~qiskit.quantum_info.operators.Clifford` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        num_qubits = 4
        qc = QuantumCircuit(4, 0)
        qc.h(2)
        qc.cx(0, 1)
        op = Clifford(qc)
        self.assertTrue(op.name == 'clifford')
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_cnotdihedral_as_operation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can instantiate an object of class\n        :class:`~qiskit.quantum_info.operators.CNOTDihedral` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        num_qubits = 4
        qc = QuantumCircuit(4)
        qc.t(0)
        qc.x(0)
        qc.t(0)
        op = CNOTDihedral(qc)
        self.assertTrue(op.name == 'cnotdihedral')
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)

    def test_pauli_as_operation(self):
        if False:
            i = 10
            return i + 15
        'Test that we can instantiate an object of class\n        :class:`~qiskit.quantum_info.operators.Pauli` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        num_qubits = 4
        op = Pauli('I' * num_qubits)
        self.assertTrue(op.name == 'pauli')
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)

    def test_isometry_as_operation(self):
        if False:
            return 10
        'Test that we can instantiate an object of class\n        :class:`~qiskit.extensions.quantum_initializer.Isometry` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        op = Isometry(np.eye(4, 4), 3, 2)
        self.assertTrue(op.name == 'isometry')
        self.assertTrue(op.num_qubits == 7)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_initialize_as_operation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can instantiate an object of class\n        :class:`~qiskit.extensions.quantum_initializer.Initialize` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        op = Initialize(desired_vector)
        self.assertTrue(op.name == 'initialize')
        self.assertTrue(op.num_qubits == 2)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_gate_as_operation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can instantiate an object of class\n        :class:`~qiskit.circuit.Gate` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        name = 'test_gate_name'
        num_qubits = 3
        op = Gate(name, num_qubits, [])
        self.assertTrue(op.name == name)
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_xgate_as_operation(self):
        if False:
            i = 10
            return i + 15
        'Test that we can instantiate an object of class\n        :class:`~qiskit.circuit.library.XGate` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        op = XGate()
        self.assertTrue(op.name == 'x')
        self.assertTrue(op.num_qubits == 1)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_cxgate_as_operation(self):
        if False:
            print('Hello World!')
        'Test that we can instantiate an object of class\n        :class:`~qiskit.circuit.library.CXGate` and that\n        it has the expected name, num_qubits and num_clbits.\n        '
        op = CXGate()
        self.assertTrue(op.name == 'cx')
        self.assertTrue(op.num_qubits == 2)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_can_append_to_quantum_circuit(self):
        if False:
            i = 10
            return i + 15
        'Test that we can add various objects with Operation interface to a Quantum Circuit.'
        qc = QuantumCircuit(6, 1)
        qc.append(XGate(), [2])
        qc.append(Barrier(3), [1, 2, 4])
        qc.append(CXGate(), [0, 1])
        qc.append(Measure(), [1], [0])
        qc.append(Reset(), [0])
        qc.cx(3, 4)
        qc.append(Gate('some_gate', 3, []), [1, 2, 3])
        qc.append(Initialize([0.5, 0.5, 0.5, 0.5]), [4, 5])
        qc.append(Isometry(np.eye(4, 4), 0, 0), [3, 4])
        qc.append(Pauli('II'), [0, 1])
        circ1 = QuantumCircuit(2)
        circ1.h(1)
        circ1.cx(0, 1)
        qc.append(Clifford(circ1), [0, 1])
        circ2 = QuantumCircuit(2)
        circ2.t(0)
        circ2.x(0)
        circ2.t(1)
        qc.append(CNOTDihedral(circ2), [2, 3])
        self.assertIsInstance(qc, QuantumCircuit)
if __name__ == '__main__':
    unittest.main()