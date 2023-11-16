import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveBarrierAPI(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        pass

    def test_barrier_nccl(self):
        if False:
            print('Hello World!')
        self.check_with_place('collective_barrier_api.py', 'barrier', 'nccl')

    def test_barrier_nccl_with_new_comm(self):
        if False:
            return 10
        self.check_with_place('collective_barrier_api.py', 'barrier', 'nccl', need_envs={'FLAGS_dynamic_static_unified_comm': 'true'})

    def test_barrier_gloo(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('collective_barrier_api.py', 'barrier', 'gloo', '5')
if __name__ == '__main__':
    unittest.main()