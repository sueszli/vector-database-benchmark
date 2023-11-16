"""Test library of integer comparison circuits."""
import unittest
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test.base import QiskitTestCase
from qiskit import BasicAer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import IntegerComparator

@ddt
class TestIntegerComparator(QiskitTestCase):
    """Test the integer comparator circuit."""

    def assertComparisonIsCorrect(self, comp, num_state_qubits, value, geq):
        if False:
            print('Hello World!')
        'Assert that the comparator output is correct.'
        qc = QuantumCircuit(comp.num_qubits)
        qc.h(list(range(num_state_qubits)))
        qc.append(comp, list(range(comp.num_qubits)))
        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(qc, backend).result().get_statevector()
        for (i, amplitude) in enumerate(statevector):
            prob = np.abs(amplitude) ** 2
            if prob > 1e-06:
                self.assertEqual(True, np.isclose(1.0, prob * 2.0 ** num_state_qubits))
                b_value = f'{i:b}'.rjust(qc.width(), '0')
                x = int(b_value[-num_state_qubits:], 2)
                comp_result = int(b_value[-num_state_qubits - 1], 2)
                if geq:
                    self.assertEqual(x >= value, comp_result == 1)
                else:
                    self.assertEqual(x < value, comp_result == 1)

    @data([1, 0, True], [1, 1, True], [2, -1, True], [3, 5, True], [3, 2, True], [3, 2, False], [4, 6, False])
    @unpack
    def test_fixed_value_comparator(self, num_state_qubits, value, geq):
        if False:
            return 10
        'Test the fixed value comparator circuit.'
        comp = IntegerComparator(num_state_qubits, value, geq=geq)
        self.assertComparisonIsCorrect(comp, num_state_qubits, value, geq)

    def test_mutability(self):
        if False:
            while True:
                i = 10
        'Test changing the arguments of the comparator.'
        comp = IntegerComparator()
        with self.subTest(msg='missing num state qubits and value'):
            with self.assertRaises(AttributeError):
                _ = str(comp.draw())
        comp.num_state_qubits = 2
        with self.subTest(msg='missing value'):
            with self.assertRaises(AttributeError):
                _ = str(comp.draw())
        comp.value = 0
        comp.geq = True
        with self.subTest(msg='updating num state qubits'):
            comp.num_state_qubits = 1
            self.assertComparisonIsCorrect(comp, 1, 0, True)
        with self.subTest(msg='updating the value'):
            comp.num_state_qubits = 3
            comp.value = 2
            self.assertComparisonIsCorrect(comp, 3, 2, True)
        with self.subTest(msg='updating geq'):
            comp.geq = False
            self.assertComparisonIsCorrect(comp, 3, 2, False)
if __name__ == '__main__':
    unittest.main()