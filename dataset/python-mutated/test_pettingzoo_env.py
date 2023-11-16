from numpy import float32
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.mpe import simple_spread_v3
from supersuit import color_reduction_v0, dtype_v0, normalize_obs_v0, observation_lambda_v0, resize_v1
from supersuit.utils.convert_box import convert_box
import unittest
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

def change_observation(obs, obs_space):
    if False:
        i = 10
        return i + 15
    obs = obs[..., None]
    return obs

def change_obs_space(obs_space):
    if False:
        print('Hello World!')
    return convert_box(lambda obs: change_observation(obs, obs_space), obs_space)

class TestPettingZooEnv(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        ray.init()

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_pettingzoo_pistonball_v6_policies_are_dict_env(self):
        if False:
            return 10

        def env_creator(config):
            if False:
                for i in range(10):
                    print('nop')
            env = pistonball_v6.env()
            env = dtype_v0(env, dtype=float32)
            env = color_reduction_v0(env, mode='R')
            env = normalize_obs_v0(env)
            env = observation_lambda_v0(env, change_observation, change_obs_space)
            env = resize_v1(env, x_size=84, y_size=84, linear_interp=True)
            return env
        register_env('pistonball', lambda config: PettingZooEnv(env_creator(config)))
        config = PPOConfig().environment('pistonball', env_config={'local_ratio': 0.5}).multi_agent(policies={'av'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'av').debugging(log_level='DEBUG').rollouts(num_rollout_workers=1, rollout_fragment_length=30).training(train_batch_size=200)
        algo = config.build()
        algo.train()
        algo.stop()

    def test_pettingzoo_env(self):
        if False:
            for i in range(10):
                print('nop')
        register_env('simple_spread', lambda _: PettingZooEnv(simple_spread_v3.env()))
        config = PPOConfig().environment('simple_spread').rollouts(num_rollout_workers=0, rollout_fragment_length=30).debugging(log_level='DEBUG').training(train_batch_size=200).multi_agent(policies={'av'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'av')
        algo = config.build()
        algo.train()
        algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))