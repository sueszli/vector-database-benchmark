import unittest
import test_communication_api_base as test_base

class TestCommunicationStreamReduceScatterAPI(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'backend': 'nccl', 'shape': '(100, 200)', 'dtype': 'float32', 'seeds': str(self._seeds)}
        self._changeable_envs = {'sync_op': ['True', 'False'], 'use_calc_stream': ['True', 'False']}

    def test_reduce_scatter_stream(self):
        if False:
            i = 10
            return i + 15
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            if eval(envs['use_calc_stream']) and (not eval(envs['sync_op'])):
                continue
            self.run_test_case('communication_stream_reduce_scatter_api_dygraph.py', user_defined_envs=envs)

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
if __name__ == '__main__':
    unittest.main()