import unittest
import collective.test_communication_api_base as test_base

class TestSemiAutoParallelDPMPStrategy(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp(num_of_devices=4, timeout=120, nnode=1)
        self._default_envs = {'dtype': 'float32', 'seed': '2023'}
        self._changeable_envs = {'backend': ['gpu']}

    def test_simple_net_bybrid_strategy(self):
        if False:
            while True:
                i = 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_dp_mp.py', user_defined_envs=envs)

class TestSemiAutoParallelHybridStrategy(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp(num_of_devices=8, timeout=120, nnode=1)
        self._default_envs = {'dtype': 'float32', 'seed': '2023'}
        self._changeable_envs = {'backend': ['gpu']}

    def test_simple_net_bybrid_strategy(self):
        if False:
            print('Hello World!')
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_dp_mp_pp.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()