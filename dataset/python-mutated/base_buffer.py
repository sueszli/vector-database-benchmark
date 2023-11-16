from typing import Union, Dict, Any, List
from abc import ABC, abstractmethod
import copy
from easydict import EasyDict
from ding.utils import import_module, BUFFER_REGISTRY

class IBuffer(ABC):
    """
    Overview:
        Buffer interface
    Interfaces:
        default_config, push, update, sample, clear, count, state_dict, load_state_dict
    """

    @classmethod
    def default_config(cls) -> EasyDict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Default config of this buffer class.\n        Returns:\n            - default_config (:obj:`EasyDict`)\n        '
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def push(self, data: Union[List[Any], Any], cur_collector_envstep: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Push a data into buffer.\n        Arguments:\n            - data (:obj:`Union[List[Any], Any]`): The data which will be pushed into buffer. Can be one \\\n                (in `Any` type), or many(int `List[Any]` type).\n            - cur_collector_envstep (:obj:`int`): Collector's current env step.\n        "
        raise NotImplementedError

    @abstractmethod
    def update(self, info: Dict[str, list]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Update data info, e.g. priority.\n        Arguments:\n            - info (:obj:`Dict[str, list]`): Info dict. Keys depends on the specific buffer type.\n        '
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, cur_learner_iter: int) -> list:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Sample data with length ``batch_size``.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled.\n            - cur_learner_iter (:obj:`int`): Learner's current iteration.\n        Returns:\n            - sampled_data (:obj:`list`): A list of data with length `batch_size`.\n        "
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Clear all the data and reset the related variables.\n        '
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        if False:
            return 10
        '\n        Overview:\n            Count how many valid datas there are in the buffer.\n        Returns:\n            - count (:obj:`int`): Number of valid data.\n        '
        raise NotImplementedError

    @abstractmethod
    def save_data(self, file_name: str):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Save buffer data into a file.\n        Arguments:\n            - file_name (:obj:`str`): file name of buffer data\n        '
        raise NotImplementedError

    @abstractmethod
    def load_data(self, file_name: str):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Load buffer data from a file.\n        Arguments:\n            - file_name (:obj:`str`): file name of buffer data\n        '
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Provide a state dict to keep a record of current buffer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.                 With the dict, one can easily reproduce the buffer.\n        '
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, _state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Load state dict to reproduce the buffer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.\n        '
        raise NotImplementedError

def create_buffer(cfg: EasyDict, *args, **kwargs) -> IBuffer:
    if False:
        return 10
    '\n    Overview:\n        Create a buffer according to cfg and other arguments.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Buffer config.\n    ArgumentsKeys:\n        - necessary: `type`\n    '
    import_module(cfg.get('import_names', []))
    if cfg.type == 'naive':
        kwargs.pop('tb_logger', None)
    return BUFFER_REGISTRY.build(cfg.type, cfg, *args, **kwargs)

def get_buffer_cls(cfg: EasyDict) -> type:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Get a buffer class according to cfg.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Buffer config.\n    ArgumentsKeys:\n        - necessary: `type`\n    '
    import_module(cfg.get('import_names', []))
    return BUFFER_REGISTRY.get(cfg.type)