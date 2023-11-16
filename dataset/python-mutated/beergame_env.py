import numpy as np
from dizoo.beergame.envs.beergame_core import BeerGame
from typing import Union, List, Optional
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray
import copy

@ENV_REGISTRY.register('beergame')
class BeerGameEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        if False:
            return 10
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        if not self._init_flag:
            self._env = BeerGame(self._cfg.role, self._cfg.agent_type, self._cfg.demandDistribution)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = self._env.reward_space
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._eval_episode_return = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if False:
            while True:
                i = 10
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if False:
            while True:
                i = 10
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.squeeze()
        (obs, rew, done, info) = self._env.step(action)
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def reward_shaping(self, transitions: List[dict]) -> List[dict]:
        if False:
            for i in range(10):
                print('nop')
        new_transitions = copy.deepcopy(transitions)
        for trans in new_transitions:
            trans['reward'] = self._env.reward_shaping(trans['reward'])
        return new_transitions

    def random_action(self) -> np.ndarray:
        if False:
            print('Hello World!')
        random_action = self.action_space.sample()
        if isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def enable_save_figure(self, figure_path: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        self._env.enable_save_figure(figure_path)

    @property
    def observation_space(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._observation_space

    @property
    def action_space(self) -> int:
        if False:
            while True:
                i = 10
        return self._action_space

    @property
    def reward_space(self) -> int:
        if False:
            return 10
        return self._reward_space

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'DI-engine Beergame Env'