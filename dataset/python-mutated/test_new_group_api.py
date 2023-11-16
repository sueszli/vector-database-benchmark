import unittest
from test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveAllreduceAPI(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_allreduce_nccl(self):
        if False:
            print('Hello World!')
        self.check_with_place('collective_allreduce_new_group_api.py', 'allreduce', 'nccl')
if __name__ == '__main__':
    unittest.main()