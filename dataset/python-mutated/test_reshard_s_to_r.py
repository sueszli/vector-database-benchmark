import unittest
import collective.test_communication_api_base as test_base

class TestReshardSToR(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'shape': '(10, 20)', 'dtype': 'float32', 'seeds': str(self._seeds)}
        self._changeable_envs = {'shard': ['0', '1'], 'backend': ['cpu', 'gpu']}

    def test_reshard_s_to_r(self):
        if False:
            while True:
                i = 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('reshard_s_to_r.py', user_defined_envs=envs)

    def test_reshard_s_to_r_cross_mesh(self):
        if False:
            for i in range(10):
                print('nop')
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            if envs['backend'] != 'cpu':
                self.run_test_case('reshard_s_to_r_cross_mesh.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()