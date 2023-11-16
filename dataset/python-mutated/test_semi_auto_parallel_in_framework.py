import unittest
import collective.test_communication_api_base as test_base

class TestSemiAutoParallelInFramework(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'dtype': 'float32', 'seed': '2023'}
        self._changeable_envs = {'backend': ['cpu', 'gpu']}

    def test_simple_net_single_strategy_with_gradient_hook(self):
        if False:
            print('Hello World!')
        self._changeable_envs = {'backend': ['gpu']}
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_gradient_hook.py', user_defined_envs=envs)

    def test_simple_net_clear_gradient(self):
        if False:
            i = 10
            return i + 15
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_clear_gradient.py', user_defined_envs=envs)

    def test_simple_net_several_grad_api(self):
        if False:
            for i in range(10):
                print('nop')
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_grad_api.py', user_defined_envs=envs)

    def test_simple_net_empty_grad(self):
        if False:
            i = 10
            return i + 15
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_fill_zero_for_emtpy_grad.py', user_defined_envs=envs)

    def test_simple_net_zero_grads(self):
        if False:
            return 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_zero_grads.py', user_defined_envs=envs)

    def test_simple_net_custom_relu(self):
        if False:
            i = 10
            return i + 15
        self._changeable_envs = {'backend': ['gpu']}
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_parallel_simple_net_custom_relu.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()