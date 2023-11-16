from typing import Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from collections import namedtuple, deque
from easydict import EasyDict
import copy
import numpy as np
import torch
from ding.utils import SERIAL_EVALUATOR_REGISTRY, import_module, lists_to_dicts
from ding.torch_utils import to_tensor, to_ndarray, tensor_to_list

class ISerialEvaluator(ABC):
    """
    Overview:
        Basic interface class for serial evaluator.
    Interfaces:
        reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Get evaluator's default config. We merge evaluator's default config with other default configs                and user's config to get the final config.\n        Return:\n            cfg: (:obj:`EasyDict`): evaluator's default config\n        "
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def reset_env(self, _env: Optional[Any]=None) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def reset_policy(self, _policy: Optional[namedtuple]=None) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def reset(self, _policy: Optional[namedtuple]=None, _env: Optional[Any]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def should_eval(self, train_iter: int) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def eval(self, save_ckpt_fn: Callable=None, train_iter: int=-1, envstep: int=-1, n_episode: Optional[int]=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

def create_serial_evaluator(cfg: EasyDict, **kwargs) -> ISerialEvaluator:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Create a specific evaluator instance based on the config.\n    '
    import_module(cfg.get('import_names', []))
    if 'type' not in cfg:
        cfg.type = 'interaction'
    return SERIAL_EVALUATOR_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)

class VectorEvalMonitor(object):
    """
    Overview:
        In some cases,  different environment in evaluator may collect different length episode. For example,             suppose we want to collect 12 episodes in evaluator but only have 5 environments, if we didn’t do             any thing, it is likely that we will get more short episodes than long episodes. As a result,             our average reward will have a bias and may not be accurate. we use VectorEvalMonitor to solve the problem.
    Interfaces:
        __init__, is_finished, update_info, update_reward, get_episode_return, get_latest_reward, get_current_episode,            get_episode_info
    """

    def __init__(self, env_num: int, n_episode: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Init method. According to the number of episodes and the number of environments, determine how many                 episodes need to be opened for each environment, and initialize the reward, info and other                 information\n        Arguments:\n            - env_num (:obj:`int`): the number of episodes need to be open\n            - n_episode (:obj:`int`): the number of environments\n        '
        assert n_episode >= env_num, 'n_episode < env_num, please decrease the number of eval env'
        self._env_num = env_num
        self._n_episode = n_episode
        each_env_episode = [n_episode // env_num for _ in range(env_num)]
        for i in range(n_episode % env_num):
            each_env_episode[i] += 1
        self._video = {env_id: deque([[] for _ in range(maxlen)], maxlen=maxlen) for (env_id, maxlen) in enumerate(each_env_episode)}
        self._reward = {env_id: deque(maxlen=maxlen) for (env_id, maxlen) in enumerate(each_env_episode)}
        self._info = {env_id: deque(maxlen=maxlen) for (env_id, maxlen) in enumerate(each_env_episode)}

    def is_finished(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Determine whether the evaluator has completed the work.\n        Return:\n            - result: (:obj:`bool`): whether the evaluator has completed the work\n        '
        return all([len(v) == v.maxlen for v in self._reward.values()])

    def update_info(self, env_id: int, info: Any) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Update the information of the environment indicated by env_id.\n        Arguments:\n            - env_id: (:obj:`int`): the id of the environment we need to update information\n            - info: (:obj:`Any`): the information we need to update\n        '
        info = tensor_to_list(info)
        self._info[env_id].append(info)

    def update_reward(self, env_id: int, reward: Any) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Update the reward indicated by env_id.\n        Arguments:\n            - env_id: (:obj:`int`): the id of the environment we need to update the reward\n            - reward: (:obj:`Any`): the reward we need to update\n        '
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self._reward[env_id].append(reward)

    def update_video(self, imgs):
        if False:
            i = 10
            return i + 15
        for (env_id, img) in imgs.items():
            if len(self._reward[env_id]) == self._reward[env_id].maxlen:
                continue
            self._video[env_id][len(self._reward[env_id])].append(img)

    def get_video(self):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Convert list of videos into [N, T, C, H, W] tensor, containing\n            worst, median, best evaluation trajectories for video logging.\n        '
        videos = sum([list(v) for v in self._video.values()], [])
        videos = [np.transpose(np.stack(video, 0), [0, 3, 1, 2]) for video in videos]
        sortarg = np.argsort(self.get_episode_return())
        if len(sortarg) == 1:
            idxs = [sortarg[0]]
        elif len(sortarg) == 2:
            idxs = [sortarg[0], sortarg[-1]]
        elif len(sortarg) == 3:
            idxs = [sortarg[0], sortarg[len(sortarg) // 2], sortarg[-1]]
        else:
            idxs = [sortarg[0], sortarg[len(sortarg) // 2 - 1], sortarg[len(sortarg) // 2], sortarg[-1]]
        videos = [videos[idx] for idx in idxs]
        max_length = max((video.shape[0] for video in videos))
        for i in range(len(videos)):
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)
        videos = np.stack(videos, 0)
        assert len(videos.shape) == 5, 'Need [N, T, C, H, W] input tensor for video logging!'
        return videos

    def get_episode_return(self) -> list:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Sum up all reward and get the total return of one episode.\n        '
        return sum([list(v) for v in self._reward.values()], [])

    def get_latest_reward(self, env_id: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get the latest reward of a certain environment.\n        Arguments:\n            - env_id: (:obj:`int`): the id of the environment we need to get reward.\n        '
        return self._reward[env_id][-1]

    def get_current_episode(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get the current episode. We can know which episode our evaluator is executing now.\n        '
        return sum([len(v) for v in self._reward.values()])

    def get_episode_info(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Get all episode information, such as total return of one episode.\n        '
        if len(self._info[0]) == 0:
            return None
        else:
            total_info = sum([list(v) for v in self._info.values()], [])
            total_info = lists_to_dicts(total_info)
            new_dict = {}
            for k in total_info.keys():
                if np.isscalar(total_info[k][0]):
                    new_dict[k + '_mean'] = np.mean(total_info[k])
            total_info.update(new_dict)
            return total_info