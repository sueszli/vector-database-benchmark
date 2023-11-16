import unittest
import collective.test_communication_api_base as test_base

class TestDistTensorAPI(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'shape': '(10, 20)', 'dtype': 'float32', 'seeds': str(self._seeds), 'shard': '0'}
        self._changeable_envs = {'backend': ['cpu', 'gpu']}

    def test_dist_tensor_api(self):
        if False:
            for i in range(10):
                print('nop')
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('semi_auto_placements.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()