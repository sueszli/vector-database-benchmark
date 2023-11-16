import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestRowParallelLinearAPI(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_row_parallel_linear(self):
        if False:
            print('Hello World!')
        self.check_with_place('row_parallel_linear_api.py', 'row_parallel_linear', 'nccl')
if __name__ == '__main__':
    unittest.main()