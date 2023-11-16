from collections import namedtuple
import numpy as np
from ding.envs.common import EnvElement

class GfootballReward(EnvElement):
    _name = 'gfootballReward'
    _reward_keys = ['reward_value']
    Reward = namedtuple('Action', _reward_keys)
    MinReward = -1.0
    MaxReward = 1.0

    def _init(self, cfg) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._default_val = 0.0
        self.template = {'reward_value': {'name': 'reward_value', 'shape': (1,), 'value': {'min': -1.0, 'max': 1.0, 'dtype': float, 'dinfo': 'float value'}, 'env_value': 'reward of action', 'to_agent_processor': lambda x: x, 'from_agent_processor': lambda x: x, 'necessary': True}}
        self._shape = (1,)
        self._value = {'min': -1.0, 'max': 1.0, 'dtype': float, 'dinfo': 'float value'}

    def _to_agent_processor(self, reward: float) -> np.array:
        if False:
            for i in range(10):
                print('nop')
        return np.array([reward], dtype=float)

    def _from_agent_processor(self, reward: float) -> float:
        if False:
            i = 10
            return i + 15
        return reward

    def _details(self):
        if False:
            i = 10
            return i + 15
        return '\t'.join(self._reward_keys)