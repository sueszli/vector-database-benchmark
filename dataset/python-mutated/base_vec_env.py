import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import cloudpickle
import gymnasium as gym
import numpy as np
from gymnasium import spaces
VecEnvIndices = Union[None, int, Iterable[int]]
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]

def tile_images(images_nhwc: Sequence[np.ndarray]) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Tile N images into one big PxQ image\n    (P,Q) are chosen to be as close as possible, and if N\n    is square, then P=Q.\n\n    :param images_nhwc: list or array of images, ndim=4 once turned into array.\n        n = batch index, h = height, w = width, c = channel\n    :return: img_HWc, ndim=3\n    '
    img_nhwc = np.asarray(images_nhwc)
    (n_images, height, width, n_channels) = img_nhwc.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: Number of environments
    :param observation_space: Observation space
    :param action_space: Action space
    """

    def __init__(self, num_envs: int, observation_space: spaces.Space, action_space: spaces.Space):
        if False:
            for i in range(10):
                print('nop')
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._seeds: List[Optional[int]] = [None for _ in range(num_envs)]
        self._options: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        try:
            render_modes = self.get_attr('render_mode')
        except AttributeError:
            warnings.warn('The `render_mode` attribute is not defined in your environment. It will be set to None.')
            render_modes = [None for _ in range(num_envs)]
        assert all((render_mode == render_modes[0] for render_mode in render_modes)), 'render_mode mode should be the same for all environments'
        self.render_mode = render_modes[0]
        render_modes = []
        if self.render_mode is not None:
            if self.render_mode == 'rgb_array':
                render_modes = ['human', 'rgb_array']
            else:
                render_modes = [self.render_mode]
        self.metadata = {'render_modes': render_modes}

    def _reset_seeds(self) -> None:
        if False:
            return 10
        '\n        Reset the seeds that are going to be used at the next reset.\n        '
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset the options that are going to be used at the next reset.\n        '
        self._options = [{} for _ in range(self.num_envs)]

    @abstractmethod
    def reset(self) -> VecEnvObs:
        if False:
            i = 10
            return i + 15
        '\n        Reset all the environments and return an array of\n        observations, or a tuple of observation arrays.\n\n        If step_async is still doing work, that work will\n        be cancelled and step_wait() should not be called\n        until step_async() is invoked again.\n\n        :return: observation\n        '
        raise NotImplementedError()

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        if False:
            while True:
                i = 10
        '\n        Tell all the environments to start taking a step\n        with the given actions.\n        Call step_wait() to get the results of the step.\n\n        You should not call this if a step_async run is\n        already pending.\n        '
        raise NotImplementedError()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        if False:
            i = 10
            return i + 15
        '\n        Wait for the step taken with step_async().\n\n        :return: observation, reward, done, information\n        '
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        if False:
            return 10
        "\n        Clean up the environment's resources.\n        "
        raise NotImplementedError()

    @abstractmethod
    def get_attr(self, attr_name: str, indices: VecEnvIndices=None) -> List[Any]:
        if False:
            i = 10
            return i + 15
        "\n        Return attribute from vectorized environment.\n\n        :param attr_name: The name of the attribute whose value to return\n        :param indices: Indices of envs to get attribute from\n        :return: List of values of 'attr_name' in all environments\n        "
        raise NotImplementedError()

    @abstractmethod
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Set attribute inside vectorized environments.\n\n        :param attr_name: The name of attribute to assign new value\n        :param value: Value to assign to `attr_name`\n        :param indices: Indices of envs to assign value\n        :return:\n        '
        raise NotImplementedError()

    @abstractmethod
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices=None, **method_kwargs) -> List[Any]:
        if False:
            print('Hello World!')
        "\n        Call instance methods of vectorized environments.\n\n        :param method_name: The name of the environment method to invoke.\n        :param indices: Indices of envs whose method to call\n        :param method_args: Any positional arguments to provide in the call\n        :param method_kwargs: Any keyword arguments to provide in the call\n        :return: List of items returned by the environment's method call\n        "
        raise NotImplementedError()

    @abstractmethod
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices=None) -> List[bool]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if environments are wrapped with a given wrapper.\n\n        :param method_name: The name of the environment method to invoke.\n        :param indices: Indices of envs whose method to call\n        :param method_args: Any positional arguments to provide in the call\n        :param method_kwargs: Any keyword arguments to provide in the call\n        :return: True if the env is wrapped, False otherwise, for each env queried.\n        '
        raise NotImplementedError()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        if False:
            for i in range(10):
                print('nop')
        '\n        Step the environments with the given action\n\n        :param actions: the action\n        :return: observation, reward, done, information\n        '
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if False:
            i = 10
            return i + 15
        '\n        Return RGB images from each environment when available\n        '
        raise NotImplementedError

    def render(self, mode: Optional[str]=None) -> Optional[np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gym environment rendering\n\n        :param mode: the rendering type\n        '
        if mode == 'human' and self.render_mode != mode:
            if self.render_mode != 'rgb_array':
                warnings.warn(f"You tried to render a VecEnv with mode='{mode}' but the render mode defined when initializing the environment must be 'human' or 'rgb_array', not '{self.render_mode}'.")
                return None
        elif mode and self.render_mode != mode:
            warnings.warn(f'Starting from gymnasium v0.26, render modes are determined during the initialization of the environment.\n                We allow to pass a mode argument to maintain a backwards compatible VecEnv API, but the mode ({mode})\n                has to be the same as the environment render mode ({self.render_mode}) which is not the case.')
            return None
        mode = mode or self.render_mode
        if mode is None:
            warnings.warn('You tried to call render() but no `render_mode` was passed to the env constructor.')
            return None
        if self.render_mode == 'human':
            self.env_method('render')
            return None
        if mode == 'rgb_array' or mode == 'human':
            images = self.get_images()
            bigimg = tile_images(images)
            if mode == 'human':
                import cv2
                cv2.imshow('vecenv', bigimg[:, :, ::-1])
                cv2.waitKey(1)
            else:
                return bigimg
        else:
            self.env_method('render')
        return None

    def seed(self, seed: Optional[int]=None) -> Sequence[Union[None, int]]:
        if False:
            while True:
                i = 10
        '\n        Sets the random seeds for all environments, based on a given seed.\n        Each individual environment will still get its own seed, by incrementing the given seed.\n        WARNING: since gym 0.26, those seeds will only be passed to the environment\n        at the next reset.\n\n        :param seed: The random seed. May be None for completely random seeding.\n        :return: Returns a list containing the seeds for each individual env.\n            Note that all list elements may be None, if the env does not return anything when being seeded.\n        '
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def set_options(self, options: Optional[Union[List[Dict], Dict]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set environment options for all environments.\n        If a dict is passed instead of a list, the same options will be used for all environments.\n        WARNING: Those options will only be passed to the environment at the next reset.\n\n        :param options: A dictionary of environment options to pass to each environment at the next reset.\n        '
        if options is None:
            options = {}
        if isinstance(options, dict):
            self._options = deepcopy([options] * self.num_envs)
        else:
            self._options = deepcopy(options)

    @property
    def unwrapped(self) -> 'VecEnv':
        if False:
            while True:
                i = 10
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        if False:
            return 10
        'Check if an attribute reference is being hidden in a recursive call to __getattr__\n\n        :param name: name of attribute to check for\n        :param already_found: whether this attribute has already been found in a wrapper\n        :return: name of module whose attribute is being shadowed, if any.\n        '
        if hasattr(self, name) and already_found:
            return f'{type(self).__module__}.{type(self).__name__}'
        else:
            return None

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        if False:
            while True:
                i = 10
        '\n        Convert a flexibly-typed reference to environment indices to an implied list of indices.\n\n        :param indices: refers to indices of envs.\n        :return: the implied list of indices.\n        '
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class

    :param venv: the vectorized environment to wrap
    :param observation_space: the observation space (can be None to load from venv)
    :param action_space: the action space (can be None to load from venv)
    """

    def __init__(self, venv: VecEnv, observation_space: Optional[spaces.Space]=None, action_space: Optional[spaces.Space]=None):
        if False:
            return 10
        self.venv = venv
        super().__init__(num_envs=venv.num_envs, observation_space=observation_space or venv.observation_space, action_space=action_space or venv.action_space)
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions: np.ndarray) -> None:
        if False:
            i = 10
            return i + 15
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self) -> VecEnvObs:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        if False:
            i = 10
            return i + 15
        pass

    def seed(self, seed: Optional[int]=None) -> Sequence[Union[None, int]]:
        if False:
            return 10
        return self.venv.seed(seed)

    def set_options(self, options: Optional[Union[List[Dict], Dict]]=None) -> None:
        if False:
            return 10
        return self.venv.set_options(options)

    def close(self) -> None:
        if False:
            return 10
        return self.venv.close()

    def render(self, mode: Optional[str]=None) -> Optional[np.ndarray]:
        if False:
            return 10
        return self.venv.render(mode=mode)

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if False:
            i = 10
            return i + 15
        return self.venv.get_images()

    def get_attr(self, attr_name: str, indices: VecEnvIndices=None) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices=None) -> None:
        if False:
            while True:
                i = 10
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices=None, **method_kwargs) -> List[Any]:
        if False:
            i = 10
            return i + 15
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices=None) -> List[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self.venv.env_is_wrapped(wrapper_class, indices=indices)

    def __getattr__(self, name: str) -> Any:
        if False:
            i = 10
            return i + 15
        'Find attribute from wrapped venv(s) if this wrapper does not have it.\n        Useful for accessing attributes from venvs which are wrapped with multiple wrappers\n        which have unique attributes of interest.\n        '
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f'{type(self).__module__}.{type(self).__name__}'
            error_str = f'Error: Recursive attribute lookup for {name} from {own_class} is ambiguous and hides attribute from {blocked_class}'
            raise AttributeError(error_str)
        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> Dict[str, Any]:
        if False:
            return 10
        'Get all (inherited) instance and class attributes\n\n        :return: all_attributes\n        '
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        if False:
            return 10
        'Recursively check wrappers to find attribute.\n\n        :param name: name of attribute to look for\n        :return: attribute\n        '
        all_attributes = self._get_all_attributes()
        if name in all_attributes:
            attr = getattr(self, name)
        elif hasattr(self.venv, 'getattr_recursive'):
            attr = self.venv.getattr_recursive(name)
        else:
            attr = getattr(self.venv, name)
        return attr

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'See base class.\n\n        :return: name of module whose attribute is being shadowed, if any.\n        '
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            shadowed_wrapper_class: Optional[str] = f'{type(self).__module__}.{type(self).__name__}'
        elif name in all_attributes and (not already_found):
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)
        return shadowed_wrapper_class

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        if False:
            return 10
        self.var = var

    def __getstate__(self) -> Any:
        if False:
            return 10
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        if False:
            print('Hello World!')
        self.var = cloudpickle.loads(var)