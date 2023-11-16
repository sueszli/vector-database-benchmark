import unittest
import legacy_test.test_collective_api_base as test_base

class TestCollectiveScatterObjectListAPI(test_base.TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        pass

    def test_scatter_nccl(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('collective_scatter_object_list_api_dygraph.py', 'scatter_object_list', 'nccl', static_mode='0', dtype='pyobject')

    def test_scatter_gloo_dygraph(self):
        if False:
            print('Hello World!')
        self.check_with_place('collective_scatter_object_list_api_dygraph.py', 'scatter_object_list', 'gloo', '3', static_mode='0', dtype='pyobject')
if __name__ == '__main__':
    unittest.main()