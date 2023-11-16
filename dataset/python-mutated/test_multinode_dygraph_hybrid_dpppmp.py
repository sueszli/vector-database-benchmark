import os
import unittest
from test_collective_multi_nodes import TestDistBase

class TestDYgraphHybrid(TestDistBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._trainers = 16
        self._init_env()

    def test_hybrid_dpppmp(self):
        if False:
            print('Hello World!')
        self.check_with_place('dygraph_hybrid_dpppmp.py', backend='nccl', need_envs=os.environ)

    def test_hybrid_recompute(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dygraph_hybrid_recompute.py', backend='nccl', need_envs=os.environ)

    def test_hybrid_fp16(self):
        if False:
            return 10
        self.check_with_place('dygraph_hybrid_fp16.py', backend='nccl', need_envs=os.environ)
if __name__ == '__main__':
    unittest.main()