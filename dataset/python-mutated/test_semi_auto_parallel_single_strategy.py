import unittest
import collective.test_communication_api_base as test_base

class TestSemiAutoParallelInSingleStrategy(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'dtype': 'float32', 'seed': '2023'}
        self._changeable_envs = {'backend': ['cpu', 'gpu']}

    def test_simple_net_single_strategy(self):
        if False:
            print('Hello World!')
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net.py', user_defined_envs=envs)

    def test_simple_net_single_strategy_with_amp(self):
        if False:
            while True:
                i = 10
        changeable_envs = {'backend': ['gpu'], 'use_master_grad': ['0', '1'], 'dtype': ['bfloat16', 'float16'], 'seed': ['2023']}
        envs_list = test_base.gen_product_envs_list({}, changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_amp.py', user_defined_envs=envs)

    def test_simple_net_single_strategy_with_gradient_merge(self):
        if False:
            while True:
                i = 10
        self._changeable_envs = {'backend': ['gpu']}
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_gradient_merge.py', user_defined_envs=envs)

    def test_simple_net_recompute(self):
        if False:
            for i in range(10):
                print('nop')
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_recompute.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()