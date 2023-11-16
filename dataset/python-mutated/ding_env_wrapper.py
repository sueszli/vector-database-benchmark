from typing import List, Optional, Union, Dict
from easydict import EasyDict
import gym
import gymnasium
import copy
import numpy as np
import treetensor.numpy as tnp
from ding.envs.common.common_function import affine_transform
from ding.envs.env_wrappers import create_env_wrapper
from ding.torch_utils import to_ndarray
from ding.utils import CloudPickleWrapper
from .base_env import BaseEnv, BaseEnvTimestep
from .default_wrapper import get_default_wrappers

class DingEnvWrapper(BaseEnv):
    """
     Overview:
         This is a wrapper for the BaseEnv class, used to provide a consistent environment interface.
     Interfaces:
         __init__, reset, step, close, seed, random_action, _wrap_env, __repr__, create_collector_env_cfg,
         create_evaluator_env_cfg, enable_save_replay, observation_space, action_space, reward_space, clone
     """

    def __init__(self, env: gym.Env=None, cfg: dict=None, seed_api: bool=True, caller: str='collector') -> None:
        if False:
            return 10
        "\n        Overview:\n            Initialize the DingEnvWrapper. Either an environment instance or a config to create the environment             instance should be passed in. For the former, i.e., an environment instance: The `env` parameter must not             be `None`, but should be the instance. It does not support subprocess environment manager. Thus, it is             usually used in simple environments. For the latter, i.e., a config to create an environment instance:             The `cfg` parameter must contain `env_id`.\n        Arguments:\n            - env (:obj:`gym.Env`): An environment instance to be wrapped.\n            - cfg (:obj:`dict`): The configuration dictionary to create an environment instance.\n            - seed_api (:obj:`bool`): Whether to use seed API. Defaults to True.\n            - caller (:obj:`str`): A string representing the caller of this method, including ``collector`` or                 ``evaluator``. Different caller may need different wrappers. Default is 'collector'.\n        "
        self._env = None
        self._raw_env = env
        self._cfg = cfg
        self._seed_api = seed_api
        self._caller = caller
        if self._cfg is None:
            self._cfg = {}
        self._cfg = EasyDict(self._cfg)
        if 'act_scale' not in self._cfg:
            self._cfg.act_scale = False
        if 'rew_clip' not in self._cfg:
            self._cfg.rew_clip = False
        if 'env_wrapper' not in self._cfg:
            self._cfg.env_wrapper = 'default'
        if 'env_id' not in self._cfg:
            self._cfg.env_id = None
        if env is not None:
            self._env = env
            self._wrap_env(caller)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._action_space.seed(0)
            self._reward_space = gym.spaces.Box(low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32)
            self._init_flag = True
        else:
            assert 'env_id' in self._cfg
            self._init_flag = False
            self._observation_space = None
            self._action_space = None
            self._reward_space = None
        self._replay_path = None

    def reset(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Resets the state of the environment. If the environment is not initialized, it will be created first.\n        Returns:\n            - obs (:obj:`Dict`): The new observation after reset.\n        '
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_id)
            self._wrap_env(self._caller)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32)
            self._init_flag = True
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(self._env, video_folder=self._replay_path, episode_trigger=lambda episode_id: True, name_prefix='rl-video-{}'.format(id(self)))
            self._replay_path = None
        if isinstance(self._env, gym.Env):
            if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
                np_seed = 100 * np.random.randint(1, 1000)
                if self._seed_api:
                    self._env.seed(self._seed + np_seed)
                self._action_space.seed(self._seed + np_seed)
            elif hasattr(self, '_seed'):
                if self._seed_api:
                    self._env.seed(self._seed)
                self._action_space.seed(self._seed)
            obs = self._env.reset()
        elif isinstance(self._env, gymnasium.Env):
            if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
                np_seed = 100 * np.random.randint(1, 1000)
                self._action_space.seed(self._seed + np_seed)
                obs = self._env.reset(seed=self._seed + np_seed)
            elif hasattr(self, '_seed'):
                self._action_space.seed(self._seed)
                obs = self._env.reset(seed=self._seed)
            else:
                obs = self._env.reset()
        else:
            raise RuntimeError('not support env type: {}'.format(type(self._env)))
        if self.observation_space.dtype == np.float32:
            obs = to_ndarray(obs, dtype=np.float32)
        else:
            obs = to_ndarray(obs)
        return obs

    def close(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Clean up the environment by closing and deleting it.\n            This method should be called when the environment is no longer needed.\n            Failing to call this method can lead to memory leaks.\n        '
        try:
            self._env.close()
            del self._env
        except:
            pass

    def seed(self, seed: int, dynamic_seed: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Set the seed for the environment.\n        Arguments:\n            - seed (:obj:`int`): The seed to set.\n            - dynamic_seed (:obj:`bool`): Whether to use dynamic seed, default is True.\n        '
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.int64, np.ndarray]) -> BaseEnvTimestep:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Execute the given action in the environment, and return the timestep (observation, reward, done, info).\n        Arguments:\n            - action (:obj:`Union[np.int64, np.ndarray]`): The action to execute in the environment.\n        Returns:\n            - timestep (:obj:`BaseEnvTimestep`): The timestep after the action execution.\n        '
        action = self._judge_action_type(action)
        if self._cfg.act_scale:
            action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)
        (obs, rew, done, info) = self._env.step(action)
        if self._cfg.rew_clip:
            rew = max(-10, rew)
        rew = np.float32(rew)
        if self.observation_space.dtype == np.float32:
            obs = to_ndarray(obs, dtype=np.float32)
        else:
            obs = to_ndarray(obs)
        rew = to_ndarray([rew], np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def _judge_action_type(self, action: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Ensure the action taken by the agent is of the correct type.\n            This method is used to standardize different action types to a common format.\n        Arguments:\n            - action (Union[np.ndarray, dict]): The action taken by the agent.\n        Returns:\n            - action (Union[np.ndarray, dict]): The formatted action.\n        '
        if isinstance(action, int):
            return action
        elif isinstance(action, np.int64):
            return int(action)
        elif isinstance(action, np.ndarray):
            if action.shape == ():
                action = action.item()
            elif action.shape == (1,) and action.dtype == np.int64:
                action = action.item()
            return action
        elif isinstance(action, dict):
            for (k, v) in action.items():
                action[k] = self._judge_action_type(v)
            return action
        elif isinstance(action, tnp.ndarray):
            return self._judge_action_type(action.json())
        else:
            raise TypeError('`action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(type(action), action))

    def random_action(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Return a random action from the action space of the environment.\n        Returns:\n            - action (:obj:`np.ndarray`): The random action.\n        '
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        elif isinstance(random_action, dict):
            random_action = to_ndarray(random_action)
        else:
            raise TypeError('`random_action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(type(random_action), random_action))
        return random_action

    def _wrap_env(self, caller: str='collector') -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Wrap the environment according to the configuration.\n        Arguments:\n            - caller (:obj:`str`): The caller of the environment, including ``collector`` or ``evaluator``.                 Different caller may need different wrappers. Default is 'collector'.\n        "
        wrapper_cfgs = self._cfg.env_wrapper
        if isinstance(wrapper_cfgs, str):
            wrapper_cfgs = get_default_wrappers(wrapper_cfgs, self._cfg.env_id, caller)
        self._wrapper_cfgs = wrapper_cfgs
        for wrapper in self._wrapper_cfgs:
            if isinstance(wrapper, Dict):
                self._env = create_env_wrapper(self._env, wrapper)
            else:
                self._env = wrapper(self._env)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Return the string representation of the instance.\n        Returns:\n            - str (:obj:`str`): The string representation of the instance.\n        '
        return 'DI-engine Env({}), generated by DingEnvWrapper'.format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Create a list of environment configuration for collectors based on the input configuration.\n        Arguments:\n            - cfg (:obj:`dict`): The input configuration dictionary.\n        Returns:\n            - env_cfgs (:obj:`List[dict]`): The list of environment configurations for collectors.\n        '
        actor_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Create a list of environment configuration for evaluators based on the input configuration.\n        Arguments:\n            - cfg (:obj:`dict`): The input configuration dictionary.\n        Returns:\n            - env_cfgs (:obj:`List[dict]`): The list of environment configurations for evaluators.\n        '
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def enable_save_replay(self, replay_path: Optional[str]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Enable the save replay functionality. The replay will be saved at the specified path.\n        Arguments:\n            - replay_path (:obj:`Optional[str]`): The path to save the replay, default is None.\n        '
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @property
    def observation_space(self) -> gym.spaces.Space:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Return the observation space of the wrapped environment.\n            The observation space represents the range and shape of possible observations\n            that the environment can provide to the agent.\n        Note:\n            If the data type of the observation space is float64, it's converted to float32\n            for better compatibility with most machine learning libraries.\n        Returns:\n            - observation_space (gym.spaces.Space): The observation space of the environment.\n        "
        if self._observation_space.dtype == np.float64:
            self._observation_space.dtype = np.float32
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Return the action space of the wrapped environment.\n            The action space represents the range and shape of possible actions\n            that the agent can take in the environment.\n        Returns:\n            - action_space (gym.spaces.Space): The action space of the environment.\n        '
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Return the reward space of the wrapped environment.\n            The reward space represents the range and shape of possible rewards\n            that the agent can receive as a result of its actions.\n        Returns:\n            - reward_space (gym.spaces.Space): The reward space of the environment.\n        '
        return self._reward_space

    def clone(self, caller: str='collector') -> BaseEnv:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Clone the current environment wrapper, creating a new environment with the same settings.\n        Arguments:\n            - caller (str): A string representing the caller of this method, including ``collector`` or ``evaluator``.                 Different caller may need different wrappers. Default is 'collector'.\n        Returns:\n            - DingEnvWrapper: A new instance of the environment with the same settings.\n        "
        try:
            spec = copy.deepcopy(self._raw_env.spec)
            raw_env = CloudPickleWrapper(self._raw_env)
            raw_env = copy.deepcopy(raw_env).data
            raw_env.__setattr__('spec', spec)
        except Exception:
            raw_env = self._raw_env
        return DingEnvWrapper(raw_env, self._cfg, self._seed_api, caller)