import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

class RepeatAfterMeEnv(gym.Env):
    """Env in which the observation at timestep minus n must be repeated."""

    def __init__(self, config=None):
        if False:
            while True:
                i = 10
        config = config or {}
        if config.get('continuous'):
            self.observation_space = Box(-1.0, 1.0, (2,))
        else:
            self.observation_space = Discrete(2)
        self.action_space = self.observation_space
        self.delay = config.get('repeat_delay', 1)
        self.episode_len = config.get('episode_len', 100)
        self.history = []

    def reset(self, *, seed=None, options=None):
        if False:
            print('Hello World!')
        self.history = [0] * self.delay
        return (self._next_obs(), {})

    def step(self, action):
        if False:
            print('Hello World!')
        obs = self.history[-(1 + self.delay)]
        reward = 0.0
        if isinstance(self.action_space, Box):
            reward = -np.sum(np.abs(action - obs))
        if isinstance(self.action_space, Discrete):
            reward = 1.0 if action == obs else -1.0
        done = truncated = len(self.history) > self.episode_len
        return (self._next_obs(), reward, done, truncated, {})

    def _next_obs(self):
        if False:
            while True:
                i = 10
        if isinstance(self.observation_space, Box):
            token = np.random.random(size=(2,))
        else:
            token = np.random.choice([0, 1])
        self.history.append(token)
        return token