import unittest
from test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphFleetApi(TestMultipleGpus):

    def test_dygraph_fleet_api(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('dygraph_fleet_api.py')
if __name__ == '__main__':
    unittest.main()