from abc import ABC, abstractmethod
from typing import Dict
from easydict import EasyDict
from ditk import logging
import os
import copy
from typing import Any
from ding.utils import REWARD_MODEL_REGISTRY, import_module, save_file

class BaseRewardModel(ABC):
    """
    Overview:
        the base class of reward model
    Interface:
        ``default_config``, ``estimate``, ``train``, ``clear_data``, ``collect_data``, ``load_expert_date``
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            i = 10
            return i + 15
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def estimate(self, data: list) -> Any:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            estimate reward\n        Arguments:\n            - data (:obj:`List`): the list of data used for estimation\n        Returns / Effects:\n            - This can be a side effect function which updates the reward value\n            - If this function returns, an example returned object can be reward (:obj:`Any`): the estimated reward\n        '
        raise NotImplementedError()

    @abstractmethod
    def train(self, data) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Training the reward model\n        Arguments:\n            - data (:obj:`Any`): Data used for training\n        Effects:\n            - This is mostly a side effect function which updates the reward model\n        '
        raise NotImplementedError()

    @abstractmethod
    def collect_data(self, data) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Collecting training data in designated formate or with designated transition.\n        Arguments:\n            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)\n        Returns / Effects:\n            - This can be a side effect function which updates the data attribute in ``self``\n        '
        raise NotImplementedError()

    @abstractmethod
    def clear_data(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Clearing training data.             This can be a side effect function which clears the data attribute in ``self``\n        '
        raise NotImplementedError()

    def load_expert_data(self, data) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Getting the expert data, usually used in inverse RL reward model\n        Arguments:\n            - data (:obj:`Any`): Expert data\n        Effects:\n            This is mostly a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``)\n        '
        pass

    def reward_deepcopy(self, train_data) -> Any:
        if False:
            return 10
        '\n        Overview:\n            this method deepcopy reward part in train_data, and other parts keep shallow copy\n            to avoid the reward part of train_data in the replay buffer be incorrectly modified.\n        Arguments:\n            - train_data (:obj:`List`): the List of train data in which the reward part will be operated by deepcopy.\n        '
        train_data_reward_deepcopy = [{k: copy.deepcopy(v) if k == 'reward' else v for (k, v) in sample.items()} for sample in train_data]
        return train_data_reward_deepcopy

    def state_dict(self) -> Dict:
        if False:
            i = 10
            return i + 15
        return {}

    def load_state_dict(self, _state_dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def save(self, path: str=None, name: str='best'):
        if False:
            i = 10
            return i + 15
        if path is None:
            path = self.cfg.exp_name
        path = os.path.join(path, 'reward_model', 'ckpt')
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
        path = os.path.join(path, 'ckpt_{}.pth.tar'.format(name))
        state_dict = self.state_dict()
        save_file(path, state_dict)
        logging.info('Saved reward model ckpt in {}'.format(path))

def create_reward_model(cfg: dict, device: str, tb_logger: 'SummaryWriter') -> BaseRewardModel:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Reward Estimation Model.\n    Arguments:\n        - cfg (:obj:`Dict`): Training config\n        - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"\n        - tb_logger (:obj:`str`): Logger, defaultly set as \'SummaryWriter\' for model summary\n    Returns:\n        - reward (:obj:`Any`): The reward model\n    '
    cfg = copy.deepcopy(cfg)
    if 'import_names' in cfg:
        import_module(cfg.pop('import_names'))
    if hasattr(cfg, 'reward_model'):
        reward_model_type = cfg.reward_model.pop('type')
    else:
        reward_model_type = cfg.pop('type')
    return REWARD_MODEL_REGISTRY.build(reward_model_type, cfg, device=device, tb_logger=tb_logger)

def get_reward_model_cls(cfg: EasyDict) -> type:
    if False:
        i = 10
        return i + 15
    import_module(cfg.get('import_names', []))
    return REWARD_MODEL_REGISTRY.get(cfg.type)