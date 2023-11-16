import unittest
import ray
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.execution import synchronous_parallel_sample
from ray.rllib.offline.feature_importance import FeatureImportance

class TestFeatureImportance(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ray.init()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_feat_importance_cartpole(self):
        if False:
            i = 10
            return i + 15
        config = MARWILConfig().environment('CartPole-v1').framework('torch')
        runner = config.build()
        policy = runner.workers.local_worker().get_policy()
        sample_batch = synchronous_parallel_sample(worker_set=runner.workers)
        for repeat in [1, 10]:
            evaluator = FeatureImportance(policy=policy, repeat=repeat)
            estimate = evaluator.estimate(sample_batch)
            assert all((val > 0 for val in estimate.values()))

    def test_feat_importance_estimate_on_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))