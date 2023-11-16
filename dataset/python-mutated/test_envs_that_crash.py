import unittest
import ray
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.tests.test_worker_failures import ForwardHealthCheckToEnvWorker
from ray.rllib.examples.env.cartpole_crashing import CartPoleCrashing
from ray.rllib.utils.error import EnvError

class TestEnvsThatCrash(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_crash_during_env_pre_checking(self):
        if False:
            print('Hello World!')
        'Expect the env pre-checking to fail on each worker.'
        config = PPOConfig().rollouts(num_rollout_workers=2, num_envs_per_worker=4).environment(env=CartPoleCrashing, env_config={'p_crash': 1.0, 'init_time_s': 0.5})
        self.assertRaisesRegex(ValueError, 'Simulated env crash', lambda : config.build())

    def test_crash_during_sampling(self):
        if False:
            while True:
                i = 10
        'Expect some sub-envs to fail (and not recover).'
        config = PPOConfig().rollouts(num_rollout_workers=2, num_envs_per_worker=3).environment(env=CartPoleCrashing, env_config={'p_crash': 0.2, 'init_time_s': 0.3, 'skip_env_checking': True})
        algo = config.build()
        self.assertRaisesRegex(EnvError, 'Simulated env crash', lambda : algo.train())

    def test_crash_only_one_worker_during_sampling_but_ignore(self):
        if False:
            print('Hello World!')
        'Expect some sub-envs to fail (and not recover), but ignore.'
        config = PPOConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker, num_rollout_workers=2, num_envs_per_worker=3, ignore_worker_failures=True).environment(env=CartPoleCrashing, env_config={'p_crash': 0.8, 'crash_on_worker_indices': [1], 'skip_env_checking': True})
        algo = config.build()
        algo.train()
        self.assertEqual(algo.workers.num_healthy_remote_workers(), 1)
        algo.stop()

    def test_crash_only_one_worker_during_sampling_but_recreate(self):
        if False:
            while True:
                i = 10
        'Expect some sub-envs to fail (and not recover), but re-create worker.'
        config = PGConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker, num_rollout_workers=2, rollout_fragment_length=10, num_envs_per_worker=3, recreate_failed_workers=True).training(train_batch_size=60).environment(env=CartPoleCrashing, env_config={'crash_after_n_steps': 10, 'p_crash': 1.0, 'crash_on_worker_indices': [2], 'skip_env_checking': True})
        algo = config.build()
        for _ in range(10):
            algo.train()
            self.assertEqual(algo.workers.num_healthy_remote_workers(), 1)
        algo.stop()

    def test_crash_sub_envs_during_sampling_but_restart_sub_envs(self):
        if False:
            return 10
        'Expect sub-envs to fail (and not recover), but re-start them individually.'
        config = PPOConfig().rollouts(num_rollout_workers=2, num_envs_per_worker=3, restart_failed_sub_environments=True, ignore_worker_failures=True).environment(env=CartPoleCrashing, env_config={'p_crash': 0.01, 'skip_env_checking': True})
        algo = config.build()
        for _ in range(10):
            algo.train()
            self.assertEqual(algo.workers.num_healthy_remote_workers(), 2)
        algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))