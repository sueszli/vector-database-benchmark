import unittest
import collective.test_communication_api_base as test_base

class TestReshardSameStatus(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'shape': '(6, 10, 20, 12)', 'dtype': 'float32', 'seeds': '100'}
        self._changeable_envs = {'backend': ['gpu']}

    def test_reshard_same_status(self):
        if False:
            while True:
                i = 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('reshard_same_status.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()