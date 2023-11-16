import os
import unittest
from test_collective_multi_nodes import TestDistBase

class TestDYgraphDPMode(TestDistBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._trainers = 16
        self._init_env()

    def test_col_parallel_linear(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('dygraph_hybrid_dp.py', backend='nccl', need_envs=os.environ)
if __name__ == '__main__':
    unittest.main()