import unittest
import ray
import ray.rllib.algorithms.impala as impala
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_compute_single_action, framework_iterator
(tf1, tf, tfv) = try_import_tf()

class TestIMPALAOffPolicyNess(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_impala_off_policyness(self):
        if False:
            i = 10
            return i + 15
        config = impala.ImpalaConfig().experimental(_enable_new_api_stack=True).environment('CartPole-v1').resources(num_gpus=0).rollouts(num_rollout_workers=4)
        num_iterations = 3
        num_aggregation_workers_options = [0, 1]
        for num_aggregation_workers in num_aggregation_workers_options:
            for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
                config.exploration_config = {}
                config.num_aggregation_workers = num_aggregation_workers
                print('aggregation-workers={}'.format(config.num_aggregation_workers))
                algo = config.build()
                for i in range(num_iterations):
                    algo.train()
                check_compute_single_action(algo)
                algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))