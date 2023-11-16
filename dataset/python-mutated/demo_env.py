from typing import Any, Union
import gym
import numpy as np
from ding.envs.env import BaseEnv, BaseEnvTimestep

class DemoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        if False:
            return 10
        self._closed = True
        self._observation_space = gym.spaces.Dict({'demo_dict': gym.spaces.Tuple([gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32), gym.spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)])})
        self._action_space = gym.spaces.Discrete(5)
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Space:
        if False:
            while True:
                i = 10
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        if False:
            for i in range(10):
                print('nop')
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        if False:
            i = 10
            return i + 15
        return self._reward_space

    def reset(self) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Resets the env to an initial state and returns an initial observation. Abstract Method from ``gym.Env``.\n        '
        self._step_count = 0
        self._env = 'A real environment'
        self._closed = False
        return self.observation_space.sample()

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self._closed = True

    def step(self, action: Any) -> 'BaseEnv.timestep':
        if False:
            while True:
                i = 10
        self._step_count += 1
        obs = self.observation_space.sample()
        rew = self.reward_space.sample()
        if self._step_count == 30:
            self._step_count = 0
            done = True
        else:
            done = False
        info = {}
        if done:
            info['eval_episode_return'] = self.reward_space.sample() * 30
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int) -> None:
        if False:
            print('Hello World!')
        self._seed = seed

    def random_action(self) -> Union[np.ndarray, int]:
        if False:
            return 10
        return self.action_space.sample()

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Demo Env for env_implementation_test.py'