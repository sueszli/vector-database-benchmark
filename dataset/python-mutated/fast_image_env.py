import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

class FastImageEnv(gym.Env):

    def __init__(self, config):
        if False:
            return 10
        self.zeros = np.zeros((84, 84, 4))
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 1.0, shape=(84, 84, 4), dtype=np.float32)
        self.i = 0

    def reset(self, *, seed=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        self.i = 0
        return (self.zeros, {})

    def step(self, action):
        if False:
            while True:
                i = 10
        self.i += 1
        done = truncated = self.i > 1000
        return (self.zeros, 1, done, truncated, {})