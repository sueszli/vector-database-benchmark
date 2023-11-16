import unittest
import test_collective_api_base as test_base

class TestCollectiveBroadcastObjectListAPI(test_base.TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_broadcast_nccl(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('collective_broadcast_object_list_api_dygraph.py', 'broadcast_object_list', 'nccl', static_mode='0', dtype='pyobject')

    def test_broadcast_gloo_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('collective_broadcast_object_list_api_dygraph.py', 'broadcast_object_list', 'gloo', '3', static_mode='0', dtype='pyobject')
if __name__ == '__main__':
    unittest.main()