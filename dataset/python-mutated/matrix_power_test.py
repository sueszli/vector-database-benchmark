"""Test for the matrix power integration test."""
import logging
import tempfile
import unittest
from apache_beam.examples import matrix_power

class MatrixPowerTest(unittest.TestCase):
    MATRIX_INPUT = b'\n0:1 1 1\n1:1 1 1\n2:1 1 1\n'.strip()
    VECTOR_INPUT = b'1 2 3'
    EXPONENT = 3
    EXPECTED_OUTPUT = '\n(0, 54.0)\n(1, 54.0)\n(2, 54.0)\n'.lstrip()

    def create_temp_file(self, contents):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(contents)
            return f.name

    def test_basics(self):
        if False:
            for i in range(10):
                print('nop')
        matrix_path = self.create_temp_file(self.MATRIX_INPUT)
        vector_path = self.create_temp_file(self.VECTOR_INPUT)
        matrix_power.run(('--input_matrix=%s --input_vector=%s --exponent=%d --output=%s.result' % (matrix_path, vector_path, self.EXPONENT, vector_path)).split())
        with open(vector_path + '.result-00000-of-00001') as result_file:
            results = result_file.read()
            self.assertEqual(sorted(self.EXPECTED_OUTPUT), sorted(results))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()