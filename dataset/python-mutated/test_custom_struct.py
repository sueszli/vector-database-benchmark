import re
import unittest
from example_tests import example_test

class TestCustomStruct(unittest.TestCase):

    def test_builtin_vectors(self):
        if False:
            for i in range(10):
                print('nop')
        output = example_test.run_example('custom_struct/builtin_vectors.py')
        assert re.match('Kernel output matches expected value.', output.decode('utf-8'))

    def test_packed_matrix(self):
        if False:
            i = 10
            return i + 15
        output = example_test.run_example('custom_struct/packed_matrix.py')
        assert re.match("Kernel output matches expected value for type 'float'.\\r?\\nKernel output matches expected value for type 'double'.", output.decode('utf-8'))

    def test_complex_struct(self):
        if False:
            for i in range(10):
                print('nop')
        output = example_test.run_example('custom_struct/complex_struct.py')
        assert re.match('Overall structure itemsize: \\d+ bytes\\r?\\nStructure members itemsize: \\[(\\s*\\d+){5}]\\r?\\nStructure members offsets: \\[(\\s*\\d+){5}]\\r?\\nComplex structure value:\\r?\\n\\s+\\[.*\\]\\r?\\nKernel output matches expected value.', output.decode('utf-8'))