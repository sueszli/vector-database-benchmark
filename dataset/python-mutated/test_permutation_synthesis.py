"""Test permutation synthesis functions."""
import unittest
import numpy as np
from ddt import ddt, data
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import LinearFunction, PermutationGate
from qiskit.synthesis import synth_permutation_acg
from qiskit.synthesis.permutation import synth_permutation_depth_lnn_kms, synth_permutation_basic
from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap
from qiskit.test import QiskitTestCase

@ddt
class TestPermutationSynthesis(QiskitTestCase):
    """Test the permutation synthesis functions."""

    @data(4, 5, 10, 15, 20)
    def test_get_ordered_swap(self, width):
        if False:
            print('Hello World!')
        'Test get_ordered_swap function produces correct swap list.'
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            swap_list = _get_ordered_swap(pattern)
            output = list(range(width))
            for (i, j) in swap_list:
                (output[i], output[j]) = (output[j], output[i])
            self.assertTrue(np.array_equal(pattern, output))
            self.assertLess(len(swap_list), width)

    @data(4, 5, 10, 15, 20)
    def test_synth_permutation_basic(self, width):
        if False:
            while True:
                i = 10
        'Test synth_permutation_basic function produces the correct\n        circuit.'
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_basic(pattern)
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, 'swap')
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))

    @data(4, 5, 10, 15, 20)
    def test_synth_permutation_acg(self, width):
        if False:
            print('Hello World!')
        'Test synth_permutation_acg function produces the correct\n        circuit.'
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_acg(pattern)
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, 'swap')
            self.assertLessEqual(qc.depth(), 2)
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))

    @data(4, 5, 10, 15, 20)
    def test_synth_permutation_depth_lnn_kms(self, width):
        if False:
            i = 10
            return i + 15
        'Test synth_permutation_depth_lnn_kms function produces the correct\n        circuit.'
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_depth_lnn_kms(pattern)
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, 'swap')
                q0 = qc.find_bit(instruction.qubits[0]).index
                q1 = qc.find_bit(instruction.qubits[1]).index
                dist = abs(q0 - q1)
                self.assertEqual(dist, 1)
            self.assertLessEqual(qc.depth(), width)
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))

    @data(4, 5, 6, 7)
    def test_permutation_matrix(self, width):
        if False:
            print('Hello World!')
        'Test that the unitary matrix constructed from permutation pattern\n        is correct.'
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_depth_lnn_kms(pattern)
            expected = Operator(qc)
            constructed = Operator(PermutationGate(pattern))
            self.assertEqual(expected, constructed)
if __name__ == '__main__':
    unittest.main()