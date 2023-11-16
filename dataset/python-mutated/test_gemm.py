import unittest
from example_tests import example_test

class TestGEMM(unittest.TestCase):

    def test_sgemm(self):
        if False:
            while True:
                i = 10
        example_test.run_example('gemm/sgemm.py')