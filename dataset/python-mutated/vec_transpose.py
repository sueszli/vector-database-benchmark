from copy import deepcopy
from typing import Dict, Union
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class VecTransposeImage(VecEnvWrapper):
    """
    Re-order channels, from HxWxC to CxHxW.
    It is required for PyTorch convolution layers.

    :param venv:
    :param skip: Skip this wrapper if needed as we rely on heuristic to apply it or not,
        which may result in unwanted behavior, see GH issue #671.
    """

    def __init__(self, venv: VecEnv, skip: bool=False):
        if False:
            print('Hello World!')
        assert is_image_space(venv.observation_space) or isinstance(venv.observation_space, spaces.Dict), 'The observation space must be an image or dictionary observation space'
        self.skip = skip
        if skip:
            super().__init__(venv)
            return
        if isinstance(venv.observation_space, spaces.Dict):
            self.image_space_keys = []
            observation_space = deepcopy(venv.observation_space)
            for (key, space) in observation_space.spaces.items():
                if is_image_space(space):
                    self.image_space_keys.append(key)
                    assert isinstance(space, spaces.Box)
                    observation_space.spaces[key] = self.transpose_space(space, key)
        else:
            assert isinstance(venv.observation_space, spaces.Box)
            observation_space = self.transpose_space(venv.observation_space)
        super().__init__(venv, observation_space=observation_space)

    @staticmethod
    def transpose_space(observation_space: spaces.Box, key: str='') -> spaces.Box:
        if False:
            return 10
        '\n        Transpose an observation space (re-order channels).\n\n        :param observation_space:\n        :param key: In case of dictionary space, the key of the observation space.\n        :return:\n        '
        assert is_image_space(observation_space), 'The observation space must be an image'
        assert not is_image_space_channels_first(observation_space), f'The observation space {key} must follow the channel last convention'
        (height, width, channels) = observation_space.shape
        new_shape = (channels, height, width)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    @staticmethod
    def transpose_image(image: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transpose an image or batch of images (re-order channels).\n\n        :param image:\n        :return:\n        '
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return np.transpose(image, (0, 3, 1, 2))

    def transpose_observations(self, observations: Union[np.ndarray, Dict]) -> Union[np.ndarray, Dict]:
        if False:
            return 10
        '\n        Transpose (if needed) and return new observations.\n\n        :param observations:\n        :return: Transposed observations\n        '
        if self.skip:
            return observations
        if isinstance(observations, dict):
            observations = deepcopy(observations)
            for k in self.image_space_keys:
                observations[k] = self.transpose_image(observations[k])
        else:
            observations = self.transpose_image(observations)
        return observations

    def step_wait(self) -> VecEnvStepReturn:
        if False:
            return 10
        (observations, rewards, dones, infos) = self.venv.step_wait()
        for (idx, done) in enumerate(dones):
            if not done:
                continue
            if 'terminal_observation' in infos[idx]:
                infos[idx]['terminal_observation'] = self.transpose_observations(infos[idx]['terminal_observation'])
        assert isinstance(observations, (np.ndarray, dict))
        return (self.transpose_observations(observations), rewards, dones, infos)

    def reset(self) -> Union[np.ndarray, Dict]:
        if False:
            return 10
        '\n        Reset all environments\n        '
        observations = self.venv.reset()
        assert isinstance(observations, (np.ndarray, dict))
        return self.transpose_observations(observations)

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.venv.close()