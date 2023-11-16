"""Test marginal_memory() function."""
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.result import marginal_memory

class TestMarginalMemory(QiskitTestCase):
    """Result operations methods."""

    def test_marginalize_memory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that memory marginalizes correctly.'
        memory = [hex(ii) for ii in range(8)]
        res = marginal_memory(memory, indices=[0])
        self.assertEqual(res, [bin(ii % 2)[2:] for ii in range(8)])

    def test_marginalize_memory_int(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that memory marginalizes correctly int output.'
        memory = [hex(ii) for ii in range(8)]
        res = marginal_memory(memory, indices=[0], int_return=True)
        self.assertEqual(res, [ii % 2 for ii in range(8)])

    def test_marginalize_memory_hex(self):
        if False:
            return 10
        'Test that memory marginalizes correctly hex output.'
        memory = [hex(ii) for ii in range(8)]
        res = marginal_memory(memory, indices=[0], hex_return=True)
        self.assertEqual(res, [hex(ii % 2) for ii in range(8)])

    def test_marginal_counts_result_memory_indices_None(self):
        if False:
            return 10
        'Test that a memory marginalizes correctly with indices=None.'
        memory = [hex(ii) for ii in range(8)]
        res = marginal_memory(memory, hex_return=True)
        self.assertEqual(res, memory)

    def test_marginalize_memory_in_parallel(self):
        if False:
            i = 10
            return i + 15
        'Test that memory marginalizes correctly multithreaded.'
        memory = [hex(ii) for ii in range(15)]
        res = marginal_memory(memory, indices=[0], parallel_threshold=1)
        self.assertEqual(res, [bin(ii % 2)[2:] for ii in range(15)])

    def test_error_on_multiple_return_types(self):
        if False:
            return 10
        'Test that ValueError raised if multiple return types are requested.'
        with self.assertRaises(ValueError):
            marginal_memory([], int_return=True, hex_return=True)

    def test_memory_level_0(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a single level 0 measurement data is correctly marginalized.'
        memory = np.asarray([[[-12974255.0, -28106672.0], [15848939.0, -53271096.0], [-18731048.0, -56490604.0]], [[-18346508.0, -26587824.0], [-12065728.0, -44948360.0], [14035275.0, -65373000.0]], [[12802274.0, -20436864.0], [-15967512.0, -37575556.0], [15201290.0, -65182832.0]], [[-9187660.0, -22197716.0], [-17028016.0, -49578552.0], [13526576.0, -61017756.0]], [[7006214.0, -32555228.0], [16144743.0, -33563124.0], [-23524160.0, -66919196.0]]], dtype=complex)
        result = marginal_memory(memory, [0, 2])
        expected = np.asarray([[[-12974255.0, -28106672.0], [-18731048.0, -56490604.0]], [[-18346508.0, -26587824.0], [14035275.0, -65373000.0]], [[12802274.0, -20436864.0], [15201290.0, -65182832.0]], [[-9187660.0, -22197716.0], [13526576.0, -61017756.0]], [[7006214.0, -32555228.0], [-23524160.0, -66919196.0]]], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_memory_level_0_avg(self):
        if False:
            return 10
        'Test that avg level 0 measurement data is correctly marginalized.'
        memory = np.asarray([[-1059254.375, -26266612.0], [-9012669.0, -41877468.0], [6027076.0, -54875060.0]], dtype=complex)
        result = marginal_memory(memory, [0, 2], avg_data=True)
        expected = np.asarray([[-1059254.375, -26266612.0], [6027076.0, -54875060.0]], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_memory_level_1(self):
        if False:
            print('Hello World!')
        'Test that a memory level 1 single data is correctly marginalized.'
        memory = np.array([[1j, 1.0, 0.5 + 0.5j], [0.5 + 0.5j, 1.0, 1j]], dtype=complex)
        result = marginal_memory(memory, [0, 2])
        expected = np.array([[1j, 0.5 + 0.5j], [0.5 + 0.5j, 1j]], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_memory_level_1_avg(self):
        if False:
            while True:
                i = 10
        'Test that avg memory level 1 data is correctly marginalized.'
        memory = np.array([1j, 1.0, 0.5 + 0.5j], dtype=complex)
        result = marginal_memory(memory, [0, 1])
        expected = np.array([1j, 1.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)