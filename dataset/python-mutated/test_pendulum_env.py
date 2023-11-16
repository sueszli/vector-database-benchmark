import pytest
import numpy as np
from easydict import EasyDict
from torch import rand
from dizoo.classic_control.pendulum.envs import PendulumEnv

@pytest.mark.envtest
class TestPendulumEnv:

    def test_naive(self):
        if False:
            return 10
        env = PendulumEnv(EasyDict({'act_scale': True}))
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (3,)
        for i in range(10):
            if i < 5:
                random_action = np.tanh(np.random.random(1))
            else:
                random_action = env.random_action()
            timestep = env.step(random_action)
            assert timestep.obs.shape == (3,)
            assert timestep.reward.shape == (1,)
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_discrete(self):
        if False:
            i = 10
            return i + 15
        env = PendulumEnv(EasyDict({'act_scale': True, 'continuous': False}))
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (3,)
        for i in range(10):
            if i < 5:
                random_action = np.array([env.action_space.sample()])
            else:
                random_action = env.random_action()
            timestep = env.step(random_action)
            print(env.observation_space, env.action_space, env.reward_space)
            print(timestep.reward, timestep.obs, timestep.reward)
            assert timestep.reward.shape == (1,)
            assert timestep.obs.shape == (3,)
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()