"""
StatePreparation test.
"""
import unittest
import math
import numpy as np
from ddt import ddt, data
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import StatePreparation

@ddt
class TestStatePreparation(QiskitTestCase):
    """Test initialization with StatePreparation class"""

    def test_prepare_from_label(self):
        if False:
            for i in range(10):
                print('nop')
        'Prepare state from label.'
        desired_sv = Statevector.from_label('01+-lr')
        qc = QuantumCircuit(6)
        qc.prepare_state('01+-lr', range(6))
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_prepare_from_int(self):
        if False:
            for i in range(10):
                print('nop')
        'Prepare state from int.'
        desired_sv = Statevector.from_label('110101')
        qc = QuantumCircuit(6)
        qc.prepare_state(53, range(6))
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_prepare_from_list(self):
        if False:
            return 10
        'Prepare state from list.'
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        qc.prepare_state([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_prepare_single_qubit(self):
        if False:
            for i in range(10):
                print('nop')
        'Prepare state in single qubit.'
        qreg = QuantumRegister(2)
        circuit = QuantumCircuit(qreg)
        circuit.prepare_state([1 / math.sqrt(2), 1 / math.sqrt(2)], qreg[1])
        expected = QuantumCircuit(qreg)
        expected.prepare_state([1 / math.sqrt(2), 1 / math.sqrt(2)], [qreg[1]])
        self.assertEqual(circuit, expected)

    def test_nonzero_state_incorrect(self):
        if False:
            print('Hello World!')
        'Test final state incorrect if initial state not zero'
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.prepare_state([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        actual_sv = Statevector(qc)
        self.assertFalse(desired_sv == actual_sv)

    @data(2, '11', [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
    def test_inverse(self, state):
        if False:
            while True:
                i = 10
        'Test inverse of StatePreparation'
        qc = QuantumCircuit(2)
        stateprep = StatePreparation(state)
        qc.append(stateprep, [0, 1])
        qc.append(stateprep.inverse(), [0, 1])
        self.assertTrue(np.allclose(Operator(qc).data, np.identity(2 ** qc.num_qubits)))

    def test_double_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test twice inverse of StatePreparation'
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        stateprep = StatePreparation([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc.append(stateprep.inverse().inverse(), [0, 1])
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_incompatible_state_and_qubit_args(self):
        if False:
            while True:
                i = 10
        'Test error raised if number of qubits not compatible with state arg'
        qc = QuantumCircuit(3)
        with self.assertRaises(QiskitError):
            qc.prepare_state('11')

    def test_incompatible_int_state_and_qubit_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Test error raised if number of qubits not compatible with  integer state arg'
        with self.assertRaises(QiskitError):
            stateprep = StatePreparation(5, num_qubits=2)
            stateprep.definition

    def test_int_state_and_no_qubit_args(self):
        if False:
            return 10
        'Test automatic determination of qubit number'
        stateprep = StatePreparation(5)
        self.assertEqual(stateprep.num_qubits, 3)

    def test_repeats(self):
        if False:
            print('Hello World!')
        'Test repeat function repeats correctly'
        qc = QuantumCircuit(2)
        qc.append(StatePreparation('01').repeat(2), [0, 1])
        self.assertEqual(qc.decompose().count_ops()['state_preparation'], 2)
if __name__ == '__main__':
    unittest.main()