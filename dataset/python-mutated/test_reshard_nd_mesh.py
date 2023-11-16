import unittest
import collective.test_communication_api_base as test_base

class TestReshardNdMesh(test_base.CommunicationTestDistBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp(num_of_devices=2, timeout=120)
        self._default_envs = {'shape': '(12, 20, 8, 16)', 'dtype': 'float32', 'seeds': '100'}
        self._changeable_envs = {'backend': ['gpu', 'cpu']}

    def test_reshard_nd_mesh(self):
        if False:
            return 10
        envs_list = test_base.gen_product_envs_list(self._default_envs, self._changeable_envs)
        for envs in envs_list:
            self.run_test_case('reshard_nd_mesh.py', user_defined_envs=envs)
if __name__ == '__main__':
    unittest.main()