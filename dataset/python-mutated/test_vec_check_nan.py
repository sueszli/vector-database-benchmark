import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

class NanAndInfEnv(gym.Env):
    """Custom Environment that raised NaNs and Infs"""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    @staticmethod
    def step(action):
        if False:
            return 10
        if np.all(np.array(action) > 0):
            obs = float('NaN')
        elif np.all(np.array(action) < 0):
            obs = float('inf')
        else:
            obs = 0
        return ([obs], 0.0, False, False, {})

    @staticmethod
    def reset(seed=None):
        if False:
            i = 10
            return i + 15
        return ([0.0], {})

    def render(self):
        if False:
            while True:
                i = 10
        pass

def test_check_nan():
    if False:
        print('Hello World!')
    'Test VecCheckNan Object'
    env = DummyVecEnv([NanAndInfEnv])
    env = VecCheckNan(env, raise_exception=True)
    env.step([[0]])
    with pytest.raises(ValueError):
        env.step([[float('NaN')]])
    with pytest.raises(ValueError):
        env.step([[float('inf')]])
    with pytest.raises(ValueError):
        env.step([[-1]])
    with pytest.raises(ValueError):
        env.step([[1]])
    env.step(np.array([[0, 1], [0, 1]]))
    env.reset()