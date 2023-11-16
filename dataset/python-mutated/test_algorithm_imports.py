import unittest
import ray
from ray.rllib.algorithms.registry import ALGORITHMS

class TestAlgorithmImport(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    def tearDown(self):
        if False:
            return 10
        ray.shutdown()

    def test_algo_import(self):
        if False:
            return 10
        for (name, func) in ALGORITHMS.items():
            func()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))