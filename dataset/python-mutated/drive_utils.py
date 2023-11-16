from typing import Optional, List
from gym import utils
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from easydict import EasyDict
from itertools import product
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt
from ding.utils.default_helper import deep_merge_dicts

class AAA:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.x = 0

def deep_update(original: dict, new_dict: dict, new_keys_allowed: bool=False, whitelist: Optional[List[str]]=None, override_all_if_type_changes: Optional[List[str]]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Updates original dict with values from new_dict recursively.\n\n    .. note::\n\n        If new key is introduced in new_dict, then if new_keys_allowed is not\n        True, an error will be thrown. Further, for sub-dicts, if the key is\n        in the whitelist, then new subkeys can be introduced.\n\n    Arguments:\n        - original (:obj:`dict`): Dictionary with default values.\n        - new_dict (:obj:`dict`): Dictionary with values to be updated\n        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.\n        - whitelist (Optional[List[str]]): List of keys that correspond to dict\n            values where new subkeys can be introduced. This is only at the top\n            level.\n        - override_all_if_type_changes(Optional[List[str]]): List of top level\n            keys with value=dict, for which we always simply override the\n            entire value (:obj:`dict`), if the "type" key in that value dict changes.\n    '
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []
    for (k, value) in new_dict.items():
        if k not in original and (not new_keys_allowed):
            raise RuntimeError('Unknown config parameter `{}`. Base config have: {}.'.format(k, original.keys()))
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            if k in override_all_if_type_changes and 'type' in value and ('type' in original[k]) and (value['type'] != original[k]['type']):
                original[k] = value
            elif k in whitelist:
                deep_update(original[k], value, True)
            else:
                deep_update(original[k], value, new_keys_allowed)
        else:
            original[k] = value
    return original

class BaseDriveEnv(gym.Env, utils.EzPickle):
    config = dict()

    @abstractmethod
    def __init__(self, cfg: Dict, **kwargs) -> None:
        if False:
            return 10
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        utils.EzPickle.__init__(self)

    @abstractmethod
    def step(self, action: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Run one step of the environment and return the observation dict.\n        '
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset current environment.\n        '
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Release all resources in environment and close.\n        '
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Set random seed.\n        '
        raise NotImplementedError

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            i = 10
            return i + 15
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    @abstractmethod
    def __repr__(self) -> str:
        if False:
            return 10
        raise NotImplementedError