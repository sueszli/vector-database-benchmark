import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestColParallelLinearAPI(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        pass

    def test_col_parallel_linear(self):
        if False:
            return 10
        self.check_with_place('column_parallel_linear_api.py', 'column_parallel_linear', 'nccl')
if __name__ == '__main__':
    unittest.main()