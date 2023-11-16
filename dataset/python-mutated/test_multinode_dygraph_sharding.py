import os
import unittest
from test_collective_multi_nodes import TestDistBase

class TestDYgrapShardingDP(TestDistBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._trainers = 16
        self._init_env()

    def test_hybrid_sharding_stage3(self):
        if False:
            return 10
        self.check_with_place('mn_dygraph_group_sharded_stage3.py', backend='nccl', need_envs=os.environ)
if __name__ == '__main__':
    unittest.main()