import unittest
import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.testing.single_agent_gym_env_runner import SingleAgentGymEnvRunner

class TestSingleAgentGymEnvRunner(unittest.TestCase):

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

    def test_sample(self):
        if False:
            return 10
        config = AlgorithmConfig().environment('CartPole-v1').rollouts(num_envs_per_worker=2, rollout_fragment_length=64)
        env_runner = SingleAgentGymEnvRunner(config=config)
        self.assertRaises(AssertionError, lambda : env_runner.sample(num_timesteps=10, num_episodes=10))
        for _ in range(100):
            (done_episodes, ongoing_episodes) = env_runner.sample(num_episodes=10)
            self.assertTrue(len(done_episodes + ongoing_episodes) == 10)
            assert len(ongoing_episodes) == 0
            self.assertTrue(all((e.is_done for e in done_episodes)))
        for _ in range(100):
            (done_episodes, ongoing_episodes) = env_runner.sample(num_timesteps=10)
            self.assertTrue(all((e.is_done for e in done_episodes)))
            self.assertTrue(not any((e.is_done for e in ongoing_episodes)))
        for _ in range(100):
            (done_episodes, ongoing_episodes) = env_runner.sample()
            self.assertTrue(all((e.is_done for e in done_episodes)))
            self.assertTrue(not any((e.is_done for e in ongoing_episodes)))

    def test_distributed_env_runner(self):
        if False:
            return 10
        'Tests, whether SingleAgentGymEnvRunner can be distributed.'
        remote_class = ray.remote(num_cpus=1, num_gpus=0)(SingleAgentGymEnvRunner)
        remote_worker_envs = [False, True]
        for envs_parallel in remote_worker_envs:
            config = AlgorithmConfig().environment('CartPole-v1').rollouts(num_rollout_workers=5, num_envs_per_worker=5, rollout_fragment_length=10, remote_worker_envs=envs_parallel)
            array = [remote_class.remote(config=config) for _ in range(config.num_rollout_workers)]
            results = [a.sample.remote() for a in array]
            results = ray.get(results)
            for result in results:
                (completed, ongoing) = result
                self.assertTrue(all((e.is_done for e in completed)))
                self.assertTrue(not any((e.is_done for e in ongoing)))
                self.assertEqual(sum((len(e) for e in completed + ongoing)), config.num_envs_per_worker * config.rollout_fragment_length)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))