import gymnasium as gym
from gymnasium.spaces import Discrete
import random

class RepeatInitialObsEnv(gym.Env):
    """Env in which the initial observation has to be repeated all the time.

    Runs for n steps.
    r=1 if action correct, -1 otherwise (max. R=100).
    """

    def __init__(self, episode_len=100):
        if False:
            while True:
                i = 10
        self.observation_space = Discrete(2)
        self.action_space = Discrete(2)
        self.token = None
        self.episode_len = episode_len
        self.num_steps = 0

    def reset(self, *, seed=None, options=None):
        if False:
            print('Hello World!')
        self.token = random.choice([0, 1])
        self.num_steps = 0
        return (self.token, {})

    def step(self, action):
        if False:
            i = 10
            return i + 15
        if action == self.token:
            reward = 1
        else:
            reward = -1
        self.num_steps += 1
        done = truncated = self.num_steps >= self.episode_len
        return (0, reward, done, truncated, {})