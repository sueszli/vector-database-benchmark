import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle

class TestCollectiveSelectScatterAPI(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_global_scatter_nccl(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.check_with_place('collective_global_scatter.py', 'global_scatter', 'nccl')

    def test_global_scatter_nccl_dygraph_eager(self):
        if False:
            i = 10
            return i + 15
        self.check_with_place('collective_global_scatter_dygraph.py', 'global_scatter', 'nccl', static_mode='0', eager_mode=True)

    def test_global_scatter_nccl_new_comm(self):
        if False:
            while True:
                i = 10
        self.check_with_place('collective_global_scatter.py', 'global_scatter', 'nccl', need_envs={'FLAGS_dynamic_static_unified_comm': 'true'})
if __name__ == '__main__':
    unittest.main()