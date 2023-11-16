import unittest
import collective.test_communication_api_base as test_base

class TestReshardSToS(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'shape': '(6, 20)', 'dtype': 'float32', 'seeds': str(self._seeds), 'backend': 'gpu'}
        self._changeable_envs = {'shape': ['(6, 20)', '(6, 20, 10)']}

    def test_reshard_s_to_s(self):
        if False:
            return 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('reshard_s_to_s.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()