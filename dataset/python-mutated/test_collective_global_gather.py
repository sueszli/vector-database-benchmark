import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle

class TestCollectiveGlobalGatherAPI(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_global_gather_nccl(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        self.check_with_place('collective_global_gather.py', 'global_gather', 'nccl')

    def test_global_gather_nccl_dygraph_eager(self):
        if False:
            print('Hello World!')
        self.check_with_place('collective_global_gather_dygraph.py', 'global_gather', 'nccl', static_mode='0', eager_mode=True)

    def test_global_gather_nccl_new_comm(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.check_with_place('collective_global_gather.py', 'global_gather', 'nccl', need_envs={'FLAGS_dynamic_static_unified_comm': 'true'})
if __name__ == '__main__':
    unittest.main()