import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree
from typing import Any, List, Optional, Union

@DeveloperAPI
def get_original_space(space: gym.Space) -> gym.Space:
    if False:
        i = 10
        return i + 15
    'Returns the original space of a space, if any.\n\n    This function recursively traverses the given space and returns the original space\n    at the very end of the chain.\n\n    Args:\n        space: The space to get the original space for.\n\n    Returns:\n        The original space or the given space itself if no original space is found.\n    '
    if hasattr(space, 'original_space'):
        return get_original_space(space.original_space)
    else:
        return space

@DeveloperAPI
def flatten_space(space: gym.Space) -> List[gym.Space]:
    if False:
        for i in range(10):
            print('nop')
    'Flattens a gym.Space into its primitive components.\n\n    Primitive components are any non Tuple/Dict spaces.\n\n    Args:\n        space: The gym.Space to flatten. This may be any\n            supported type (including nested Tuples and Dicts).\n\n    Returns:\n        List[gym.Space]: The flattened list of primitive Spaces. This list\n            does not contain Tuples or Dicts anymore.\n    '

    def _helper_flatten(space_, return_list):
        if False:
            i = 10
            return i + 15
        from ray.rllib.utils.spaces.flexdict import FlexDict
        if isinstance(space_, Tuple):
            for s in space_:
                _helper_flatten(s, return_list)
        elif isinstance(space_, (Dict, FlexDict)):
            for k in sorted(space_.spaces):
                _helper_flatten(space_[k], return_list)
        else:
            return_list.append(space_)
    ret = []
    _helper_flatten(space, ret)
    return ret

@DeveloperAPI
def get_base_struct_from_space(space):
    if False:
        i = 10
        return i + 15
    'Returns a Tuple/Dict Space as native (equally structured) py tuple/dict.\n\n    Args:\n        space: The Space to get the python struct for.\n\n    Returns:\n        Union[dict,tuple,gym.Space]: The struct equivalent to the given Space.\n            Note that the returned struct still contains all original\n            "primitive" Spaces (e.g. Box, Discrete).\n\n    .. testcode::\n        :skipif: True\n\n        get_base_struct_from_space(Dict({\n            "a": Box(),\n            "b": Tuple([Discrete(2), Discrete(3)])\n        }))\n\n    .. testoutput::\n\n        dict(a=Box(), b=tuple(Discrete(2), Discrete(3)))\n    '

    def _helper_struct(space_):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(space_, Tuple):
            return tuple((_helper_struct(s) for s in space_))
        elif isinstance(space_, Dict):
            return {k: _helper_struct(space_[k]) for k in space_.spaces}
        else:
            return space_
    return _helper_struct(space)

@DeveloperAPI
def get_dummy_batch_for_space(space: gym.Space, batch_size: int=32, fill_value: Union[float, int, str]=0.0, time_size: Optional[int]=None, time_major: bool=False) -> np.ndarray:
    if False:
        print('Hello World!')
    'Returns batched dummy data (using `batch_size`) for the given `space`.\n\n    Note: The returned batch will not pass a `space.contains(batch)` test\n    as an additional batch dimension has to be added as dim=0.\n\n    Args:\n        space: The space to get a dummy batch for.\n        batch_size: The required batch size (B). Note that this can also\n            be 0 (only if `time_size` is None!), which will result in a\n            non-batched sample for the given space (no batch dim).\n        fill_value: The value to fill the batch with\n            or "random" for random values.\n        time_size: If not None, add an optional time axis\n            of `time_size` size to the returned batch.\n        time_major: If True AND `time_size` is not None, return batch\n            as shape [T x B x ...], otherwise as [B x T x ...]. If `time_size`\n            if None, ignore this setting and return [B x ...].\n\n    Returns:\n        The dummy batch of size `bqtch_size` matching the given space.\n    '
    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        return tree.map_structure(lambda s: get_dummy_batch_for_space(s, batch_size, fill_value), get_base_struct_from_space(space))
    elif fill_value == 'random':
        if time_size is not None:
            assert batch_size > 0 and time_size > 0
            if time_major:
                return np.array([[space.sample() for _ in range(batch_size)] for t in range(time_size)], dtype=space.dtype)
            else:
                return np.array([[space.sample() for t in range(time_size)] for _ in range(batch_size)], dtype=space.dtype)
        else:
            return np.array([space.sample() for _ in range(batch_size)] if batch_size > 0 else space.sample(), dtype=space.dtype)
    else:
        if time_size is not None:
            assert batch_size > 0 and time_size > 0
            if time_major:
                shape = [time_size, batch_size]
            else:
                shape = [batch_size, time_size]
        else:
            shape = [batch_size] if batch_size > 0 else []
        return np.full(shape + list(space.shape), fill_value=fill_value, dtype=space.dtype)

@DeveloperAPI
def flatten_to_single_ndarray(input_):
    if False:
        return 10
    'Returns a single np.ndarray given a list/tuple of np.ndarrays.\n\n    Args:\n        input_ (Union[List[np.ndarray], np.ndarray]): The list of ndarrays or\n            a single ndarray.\n\n    Returns:\n        np.ndarray: The result after concatenating all single arrays in input_.\n\n    .. testcode::\n        :skipif: True\n\n        flatten_to_single_ndarray([\n            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),\n            np.array([7, 8, 9]),\n        ])\n\n    .. testoutput::\n\n        np.array([\n            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0\n        ])\n    '
    if isinstance(input_, (list, tuple, dict)):
        expanded = []
        for in_ in tree.flatten(input_):
            expanded.append(np.reshape(in_, [-1]))
        input_ = np.concatenate(expanded, axis=0).flatten()
    return input_

@DeveloperAPI
def unbatch(batches_struct):
    if False:
        while True:
            i = 10
    'Converts input from (nested) struct of batches to batch of structs.\n\n    Input: Struct of different batches (each batch has size=3):\n        {\n            "a": np.array([1, 2, 3]),\n            "b": (np.array([4, 5, 6]), np.array([7.0, 8.0, 9.0]))\n        }\n    Output: Batch (list) of structs (each of these structs representing a\n        single action):\n        [\n            {"a": 1, "b": (4, 7.0)},  <- action 1\n            {"a": 2, "b": (5, 8.0)},  <- action 2\n            {"a": 3, "b": (6, 9.0)},  <- action 3\n        ]\n\n    Args:\n        batches_struct: The struct of component batches. Each leaf item\n            in this struct represents the batch for a single component\n            (in case struct is tuple/dict).\n            Alternatively, `batches_struct` may also simply be a batch of\n            primitives (non tuple/dict).\n\n    Returns:\n        List[struct[components]]: The list of rows. Each item\n            in the returned list represents a single (maybe complex) struct.\n    '
    flat_batches = tree.flatten(batches_struct)
    out = []
    for batch_pos in range(len(flat_batches[0])):
        out.append(tree.unflatten_as(batches_struct, [flat_batches[i][batch_pos] for i in range(len(flat_batches))]))
    return out

@DeveloperAPI
def clip_action(action, action_space):
    if False:
        i = 10
        return i + 15
    'Clips all components in `action` according to the given Space.\n\n    Only applies to Box components within the action space.\n\n    Args:\n        action: The action to be clipped. This could be any complex\n            action, e.g. a dict or tuple.\n        action_space: The action space struct,\n            e.g. `{"a": Distrete(2)}` for a space: Dict({"a": Discrete(2)}).\n\n    Returns:\n        Any: The input action, but clipped by value according to the space\'s\n            bounds.\n    '

    def map_(a, s):
        if False:
            print('Hello World!')
        if isinstance(s, gym.spaces.Box):
            a = np.clip(a, s.low, s.high)
        return a
    return tree.map_structure(map_, action, action_space)

@DeveloperAPI
def unsquash_action(action, action_space_struct):
    if False:
        print('Hello World!')
    'Unsquashes all components in `action` according to the given Space.\n\n    Inverse of `normalize_action()`. Useful for mapping policy action\n    outputs (normalized between -1.0 and 1.0) to an env\'s action space.\n    Unsquashing results in cont. action component values between the\n    given Space\'s bounds (`low` and `high`). This only applies to Box\n    components within the action space, whose dtype is float32 or float64.\n\n    Args:\n        action: The action to be unsquashed. This could be any complex\n            action, e.g. a dict or tuple.\n        action_space_struct: The action space struct,\n            e.g. `{"a": Box()}` for a space: Dict({"a": Box()}).\n\n    Returns:\n        Any: The input action, but unsquashed, according to the space\'s\n            bounds. An unsquashed action is ready to be sent to the\n            environment (`BaseEnv.send_actions([unsquashed actions])`).\n    '

    def map_(a, s):
        if False:
            print('Hello World!')
        if isinstance(s, gym.spaces.Box) and np.all(s.bounded_below) and np.all(s.bounded_above):
            if s.dtype == np.float32 or s.dtype == np.float64:
                a = s.low + (a + 1.0) * (s.high - s.low) / 2.0
                a = np.clip(a, s.low, s.high)
            elif np.issubdtype(s.dtype, np.integer):
                a = s.low + a
        return a
    return tree.map_structure(map_, action, action_space_struct)

@DeveloperAPI
def normalize_action(action, action_space_struct):
    if False:
        print('Hello World!')
    'Normalizes all (Box) components in `action` to be in [-1.0, 1.0].\n\n    Inverse of `unsquash_action()`. Useful for mapping an env\'s action\n    (arbitrary bounded values) to a [-1.0, 1.0] interval.\n    This only applies to Box components within the action space, whose\n    dtype is float32 or float64.\n\n    Args:\n        action: The action to be normalized. This could be any complex\n            action, e.g. a dict or tuple.\n        action_space_struct: The action space struct,\n            e.g. `{"a": Box()}` for a space: Dict({"a": Box()}).\n\n    Returns:\n        Any: The input action, but normalized, according to the space\'s\n            bounds.\n    '

    def map_(a, s):
        if False:
            return 10
        if isinstance(s, gym.spaces.Box) and (s.dtype == np.float32 or s.dtype == np.float64):
            a = (a - s.low) * 2.0 / (s.high - s.low) - 1.0
        return a
    return tree.map_structure(map_, action, action_space_struct)

@DeveloperAPI
def convert_element_to_space_type(element: Any, sampled_element: Any) -> Any:
    if False:
        i = 10
        return i + 15
    'Convert all the components of the element to match the space dtypes.\n\n    Args:\n        element: The element to be converted.\n        sampled_element: An element sampled from a space to be matched\n            to.\n\n    Returns:\n        The input element, but with all its components converted to match\n        the space dtypes.\n    '

    def map_(elem, s):
        if False:
            print('Hello World!')
        if isinstance(s, np.ndarray):
            if not isinstance(elem, np.ndarray):
                assert isinstance(elem, (float, int)), f'ERROR: `elem` ({elem}) must be np.array, float or int!'
                if s.shape == ():
                    elem = np.array(elem, dtype=s.dtype)
                else:
                    raise ValueError('Element should be of type np.ndarray but is instead of                             type {}'.format(type(elem)))
            elif s.dtype != elem.dtype:
                elem = elem.astype(s.dtype)
        elif isinstance(s, int) or isinstance(s, np.int_):
            if isinstance(elem, float) and elem.is_integer():
                elem = int(elem)
            if isinstance(elem, np.float_):
                elem = np.int64(elem)
        return elem
    return tree.map_structure(map_, element, sampled_element, check_types=False)