import unittest
import test_collective_api_base as test_base

class TestCollectiveAllgatherObjectAPI(test_base.TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_allgather_nccl(self):
        if False:
            while True:
                i = 10
        self.check_with_place('collective_allgather_object_api_dygraph.py', 'allgather_object', 'nccl', static_mode='0', dtype='pyobject')

    def test_allgather_gloo_dygraph(self):
        if False:
            return 10
        self.check_with_place('collective_allgather_object_api_dygraph.py', 'allgather_object', 'gloo', '3', static_mode='0', dtype='pyobject')
if __name__ == '__main__':
    unittest.main()