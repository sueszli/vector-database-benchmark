from typing import Any, List, Union, Optional
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('mountain_car')
class MountainCarEnv(BaseEnv):
    """
    Implementation of DI-engine's version of the Mountain Car deterministic MDP. 

    Important references that contributed to the creation of this env:
    > Source code of OpenAI's mountain car gym : https://is.gd/y1FkMT
    > Gym documentation of mountain car : https://is.gd/29S0dt
    > Based off DI-engine existing implementation of cartpole_env.py
    > DI-engine's env creation conventions : https://is.gd/ZHLISj

    Only __init__ , step, seed and reset are mandatory & impt.
    The other methods are generally for convenience.
    """

    def __init__(self, cfg: EasyDict) -> None:
        if False:
            i = 10
            return i + 15
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._observation_space = gym.spaces.Box(low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), shape=(2,), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(3, start=0)
        self._reward_space = gym.spaces.Box(low=-1, high=0.0, shape=(1,), dtype=np.float32)

    def seed(self, seed: int, dynamic_seed: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> np.ndarray:
        if False:
            return 10
        if not self._init_flag:
            self._env = gym.make('MountainCar-v0')
            self._init_flag = True
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(self._env, video_folder=self._replay_path, episode_trigger=lambda episode_id: True, name_prefix='rl-video-{}'.format(id(self)))
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._action_space.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0.0
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        if False:
            print('Hello World!')
        assert isinstance(action, np.ndarray), type(action)
        action = action.squeeze()
        (obs, rew, done, info) = self._env.step(action)
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        obs = to_ndarray(obs)
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def enable_save_replay(self, replay_path: Optional[str]=None) -> None:
        if False:
            return 10
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        if False:
            return 10
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        if False:
            i = 10
            return i + 15
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        if False:
            return 10
        return self._reward_space

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'DI-engine Mountain Car Env'