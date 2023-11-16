import unittest
import collective.test_communication_api_base as test_base

class TestSemiAutoParallelInplace(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'dtype': 'float32', 'seed': '2023'}
        self._changeable_envs = {'backend': ['cpu', 'gpu']}

    def test_simple_net_single_strategy(self):
        if False:
            while True:
                i = 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_dygraph_inplace.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()