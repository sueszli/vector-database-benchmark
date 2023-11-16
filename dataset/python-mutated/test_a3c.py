import unittest
from rllib_a3c.a3c import A3CConfig
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.test_utils import check_compute_single_action, check_train_results, framework_iterator

class TestA3C(unittest.TestCase):
    """Sanity tests for A2C exec impl."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ray.init(num_cpus=4)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_a3c_compilation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test whether an A3C can be built with both frameworks.'
        config = A3CConfig().rollouts(num_rollout_workers=2, num_envs_per_worker=2)
        num_iterations = 2
        for _ in framework_iterator(config):
            config.eager_tracing = False
            for env in ['CartPole-v1', 'Pendulum-v1']:
                print('env={}'.format(env))
                config.model['use_lstm'] = env == 'CartPole-v1'
                algo = config.build(env=env)
                for i in range(num_iterations):
                    results = algo.train()
                    check_train_results(results)
                    print(results)
                check_compute_single_action(algo, include_state=config.model['use_lstm'])
                algo.stop()

    def test_a3c_entropy_coeff_schedule(self):
        if False:
            i = 10
            return i + 15
        'Test A3C entropy coeff schedule support.'
        config = A3CConfig().rollouts(num_rollout_workers=1, num_envs_per_worker=1, batch_mode='truncate_episodes', rollout_fragment_length=10)
        config.training(train_batch_size=20, entropy_coeff=0.01, entropy_coeff_schedule=[[0, 0.01], [120, 0.0001]])
        config.reporting(min_time_s_per_iteration=0, min_sample_timesteps_per_iteration=20)

        def _step_n_times(trainer, n: int):
            if False:
                for i in range(10):
                    print('nop')
            'Step trainer n times.\n\n            Returns:\n                learning rate at the end of the execution.\n            '
            for _ in range(n):
                results = trainer.train()
            return results['info'][LEARNER_INFO][DEFAULT_POLICY_ID][LEARNER_STATS_KEY]['entropy_coeff']
        for _ in framework_iterator(config, frameworks=('torch', 'tf')):
            config.eager_tracing = False
            algo = config.build(env='CartPole-v1')
            coeff = _step_n_times(algo, 1)
            self.assertGreaterEqual(coeff, 0.005)
            coeff = _step_n_times(algo, 10)
            self.assertLessEqual(coeff, 0.00011)
            algo.stop()
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))