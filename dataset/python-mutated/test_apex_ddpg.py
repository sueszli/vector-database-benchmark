import unittest
import pytest
from rllib_apex_ddpg.apex_ddpg.apex_ddpg import ApexDDPGConfig
import ray
from ray.rllib.utils.test_utils import check, check_compute_single_action, check_train_results, framework_iterator

class TestApexDDPG(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    def tearDown(self):
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_apex_ddpg_compilation_and_per_worker_epsilon_values(self):
        if False:
            for i in range(10):
                print('nop')
        'Test whether APEX-DDPG can be built on all frameworks.'
        config = ApexDDPGConfig().environment(env='Pendulum-v1').rollouts(num_rollout_workers=2).reporting(min_sample_timesteps_per_iteration=100).training(num_steps_sampled_before_learning_starts=0, optimizer={'num_replay_buffer_shards': 1})
        num_iterations = 1
        for _ in framework_iterator(config, with_eager_tracing=True):
            algo = config.build()
            infos = algo.workers.foreach_policy(lambda p, _: p.get_exploration_state())
            scale = [i['cur_scale'] for i in infos]
            expected = [0.4 ** (1 + (i + 1) / float(config.num_rollout_workers - 1) * 7) for i in range(config.num_rollout_workers)]
            check(scale, [0.0] + expected)
            for _ in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            infos = algo.workers.foreach_policy(lambda p, _: p.get_exploration_state())
            scale = [i['cur_scale'] for i in infos]
            check(scale, [0.0] + expected)
            algo.stop()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))