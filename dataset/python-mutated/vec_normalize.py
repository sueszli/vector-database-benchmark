import inspect
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
from gymnasium import spaces
from stable_baselines3.common import utils
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    :param norm_obs_keys: Which keys from observation dict to normalize.
        If not specified, all keys will be normalized.
    """
    obs_spaces: Dict[str, spaces.Space]
    old_obs: Union[np.ndarray, Dict[str, np.ndarray]]

    def __init__(self, venv: VecEnv, training: bool=True, norm_obs: bool=True, norm_reward: bool=True, clip_obs: float=10.0, clip_reward: float=10.0, gamma: float=0.99, epsilon: float=1e-08, norm_obs_keys: Optional[List[str]]=None):
        if False:
            i = 10
            return i + 15
        VecEnvWrapper.__init__(self, venv)
        self.norm_obs = norm_obs
        self.norm_obs_keys = norm_obs_keys
        if self.norm_obs:
            self._sanity_checks()
            if isinstance(self.observation_space, spaces.Dict):
                self.obs_spaces = self.observation_space.spaces
                self.obs_rms = {key: RunningMeanStd(shape=self.obs_spaces[key].shape) for key in self.norm_obs_keys}
                for key in self.obs_rms.keys():
                    if is_image_space(self.obs_spaces[key]):
                        self.observation_space.spaces[key] = spaces.Box(low=-clip_obs, high=clip_obs, shape=self.obs_spaces[key].shape, dtype=np.float32)
            else:
                self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
                if is_image_space(self.observation_space):
                    self.observation_space = spaces.Box(low=-clip_obs, high=clip_obs, shape=self.observation_space.shape, dtype=np.float32)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_reward = np.array([])

    def _sanity_checks(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check the observations that are going to be normalized are of the correct type (spaces.Box).\n        '
        if isinstance(self.observation_space, spaces.Dict):
            if self.norm_obs_keys is None:
                self.norm_obs_keys = list(self.observation_space.spaces.keys())
            for obs_key in self.norm_obs_keys:
                if not isinstance(self.observation_space.spaces[obs_key], spaces.Box):
                    raise ValueError(f'VecNormalize only supports `gym.spaces.Box` observation spaces but {obs_key} is of type {self.observation_space.spaces[obs_key]}. You should probably explicitely pass the observation keys  that should be normalized via the `norm_obs_keys` parameter.')
        elif isinstance(self.observation_space, spaces.Box):
            if self.norm_obs_keys is not None:
                raise ValueError('`norm_obs_keys` param is applicable only with `gym.spaces.Dict` observation spaces')
        else:
            raise ValueError(f'VecNormalize only supports `gym.spaces.Box` and `gym.spaces.Dict` observation spaces, not {self.observation_space}')

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "\n        Gets state for pickling.\n\n        Excludes self.venv, as in general VecEnv's may not be pickleable."
        state = self.__dict__.copy()
        del state['venv']
        del state['class_attributes']
        del state['returns']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Restores pickled state.\n\n        User must call set_venv() after unpickling before using.\n\n        :param state:'
        if 'norm_obs_keys' not in state and isinstance(state['observation_space'], spaces.Dict):
            state['norm_obs_keys'] = list(state['observation_space'].spaces.keys())
        self.__dict__.update(state)
        assert 'venv' not in state
        self.venv = None

    def set_venv(self, venv: VecEnv) -> None:
        if False:
            print('Hello World!')
        '\n        Sets the vector environment to wrap to venv.\n\n        Also sets attributes derived from this such as `num_env`.\n\n        :param venv:\n        '
        if self.venv is not None:
            raise ValueError('Trying to set venv of already initialized VecNormalize wrapper.')
        self.venv = venv
        self.num_envs = venv.num_envs
        self.class_attributes = dict(inspect.getmembers(self.__class__))
        self.render_mode = venv.render_mode
        utils.check_shape_equal(self.observation_space, venv.observation_space)
        self.returns = np.zeros(self.num_envs)

    def step_wait(self) -> VecEnvStepReturn:
        if False:
            return 10
        '\n        Apply sequence of actions to sequence of environments\n        actions -> (observations, rewards, dones)\n\n        where ``dones`` is a boolean vector indicating whether each element is new.\n        '
        (obs, rewards, dones, infos) = self.venv.step_wait()
        assert isinstance(obs, (np.ndarray, dict))
        self.old_obs = obs
        self.old_reward = rewards
        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)
        obs = self.normalize_obs(obs)
        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)
        for (idx, done) in enumerate(dones):
            if not done:
                continue
            if 'terminal_observation' in infos[idx]:
                infos[idx]['terminal_observation'] = self.normalize_obs(infos[idx]['terminal_observation'])
        self.returns[dones] = 0
        return (obs, rewards, dones, infos)

    def _update_reward(self, reward: np.ndarray) -> None:
        if False:
            while True:
                i = 10
        'Update reward normalization statistics.'
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        if False:
            return 10
        '\n        Helper to normalize observation.\n        :param obs:\n        :param obs_rms: associated statistics\n        :return: normalized observation\n        '
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Helper to unnormalize observation.\n        :param obs:\n        :param obs_rms: associated statistics\n        :return: unnormalized observation\n        '
        return obs * np.sqrt(obs_rms.var + self.epsilon) + obs_rms.mean

    def normalize_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if False:
            print('Hello World!')
        "\n        Normalize observations using this VecNormalize's observations statistics.\n        Calling this method does not update statistics.\n        "
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                assert self.norm_obs_keys is not None
                for key in self.norm_obs_keys:
                    obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(np.float32)
            else:
                assert isinstance(self.obs_rms, RunningMeanStd)
                obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)
        return obs_

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        "\n        Normalize rewards using this VecNormalize's rewards statistics.\n        Calling this method does not update statistics.\n        "
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward

    def unnormalize_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if False:
            print('Hello World!')
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                assert self.norm_obs_keys is not None
                for key in self.norm_obs_keys:
                    obs_[key] = self._unnormalize_obs(obs[key], self.obs_rms[key])
            else:
                assert isinstance(self.obs_rms, RunningMeanStd)
                obs_ = self._unnormalize_obs(obs, self.obs_rms)
        return obs_

    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if False:
            return 10
        if self.norm_reward:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon)
        return reward

    def get_original_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if False:
            return 10
        '\n        Returns an unnormalized version of the observations from the most recent\n        step or reset.\n        '
        return deepcopy(self.old_obs)

    def get_original_reward(self) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Returns an unnormalized version of the rewards from the most recent step.\n        '
        return self.old_reward.copy()

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset all environments\n        :return: first observation of the episode\n        '
        obs = self.venv.reset()
        assert isinstance(obs, (np.ndarray, dict))
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                assert isinstance(self.obs_rms, RunningMeanStd)
                self.obs_rms.update(obs)
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> 'VecNormalize':
        if False:
            print('Hello World!')
        '\n        Loads a saved VecNormalize object.\n\n        :param load_path: the path to load from.\n        :param venv: the VecEnv to wrap.\n        :return:\n        '
        with open(load_path, 'rb') as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Save current VecNormalize object with\n        all running statistics and settings (e.g. clip_obs)\n\n        :param save_path: The path to save to\n        '
        with open(save_path, 'wb') as file_handler:
            pickle.dump(self, file_handler)