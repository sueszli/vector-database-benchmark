"""Test library of quantum circuits."""
import unittest
from ddt import ddt, data, unpack
import numpy as np
from qiskit.test.base import QiskitTestCase
from qiskit.circuit.library import FourierChecking
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Operator

@ddt
class TestFourierCheckingLibrary(QiskitTestCase):
    """Test the Fourier Checking circuit."""

    def assertFourierCheckingIsCorrect(self, f_truth_table, g_truth_table, fc_circuit):
        if False:
            for i in range(10):
                print('nop')
        'Assert that the Fourier Checking circuit produces the correct matrix.'
        simulated = Operator(fc_circuit)
        num_qubits = int(np.log2(len(f_truth_table)))
        h_i = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        h_tot = np.array([1])
        for _ in range(num_qubits):
            h_tot = np.kron(h_tot, h_i)
        f_mat = np.diag(f_truth_table)
        g_mat = np.diag(g_truth_table)
        expected = np.linalg.multi_dot([h_tot, g_mat, h_tot, f_mat, h_tot])
        expected = Operator(expected)
        self.assertTrue(expected.equiv(simulated))

    @data(([1, -1, -1, -1], [1, 1, -1, -1]), ([1, 1, 1, 1], [1, 1, 1, 1]))
    @unpack
    def test_fourier_checking(self, f_truth_table, g_truth_table):
        if False:
            return 10
        'Test if the Fourier Checking circuit produces the correct matrix.'
        fc_circuit = FourierChecking(f_truth_table, g_truth_table)
        self.assertFourierCheckingIsCorrect(f_truth_table, g_truth_table, fc_circuit)

    @data(([1, -1, -1, -1], [1, 1, -1]), ([1], [-1]), ([1, -1, -1, -1, 1], [1, 1, -1, -1, 1]))
    @unpack
    def test_invalid_input_raises(self, f_truth_table, g_truth_table):
        if False:
            while True:
                i = 10
        'Test that invalid input truth tables raise an error.'
        with self.assertRaises(CircuitError):
            FourierChecking(f_truth_table, g_truth_table)
if __name__ == '__main__':
    unittest.main()