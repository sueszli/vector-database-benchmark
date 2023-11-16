from abc import ABC, abstractmethod, abstractproperty
from easydict import EasyDict
from ding.utils import EasyTimer, import_module, get_task_uid, dist_init, dist_finalize, COMM_LEARNER_REGISTRY
from ding.policy import create_policy
from ding.worker.learner import create_learner

class BaseCommLearner(ABC):
    """
    Overview:
        Abstract baseclass for CommLearner.
    Interfaces:
        __init__, send_policy, get_data, send_learn_info， start, close
    Property:
        hooks4call
    """

    def __init__(self, cfg: 'EasyDict') -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Initialization method.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Config dict\n        '
        self._cfg = cfg
        self._learner_uid = get_task_uid()
        self._timer = EasyTimer()
        if cfg.multi_gpu:
            (self._rank, self._world_size) = dist_init()
        else:
            (self._rank, self._world_size) = (0, 1)
        self._multi_gpu = cfg.multi_gpu
        self._end_flag = True

    @abstractmethod
    def send_policy(self, state_dict: dict) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Save learner's policy in corresponding path.\n            Will be registered in base learner.\n        Arguments:\n            - state_dict (:obj:`dict`): State dict of the runtime policy.\n        "
        raise NotImplementedError

    @abstractmethod
    def get_data(self, batch_size: int) -> list:
        if False:
            return 10
        '\n        Overview:\n            Get batched meta data from coordinator.\n            Will be registered in base learner.\n        Arguments:\n            - batch_size (:obj:`int`): Batch size.\n        Returns:\n            - stepdata (:obj:`list`): A list of training data, each element is one trajectory.\n        '
        raise NotImplementedError

    @abstractmethod
    def send_learn_info(self, learn_info: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Send learn info to coordinator.\n            Will be registered in base learner.\n        Arguments:\n            - learn_info (:obj:`dict`): Learn info in dict type.\n        '
        raise NotImplementedError

    def start(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Start comm learner.\n        '
        self._end_flag = False

    def close(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Close comm learner.\n        '
        self._end_flag = True
        if self._multi_gpu:
            dist_finalize()

    @abstractproperty
    def hooks4call(self) -> list:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            - hooks (:obj:`list`): The hooks which comm learner has. Will be registered in learner as well.\n        '
        raise NotImplementedError

    def _create_learner(self, task_info: dict) -> 'BaseLearner':
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Receive ``task_info`` passed from coordinator and create a learner.\n        Arguments:\n            - task_info (:obj:`dict`): Task info dict from coordinator. Should be like                 {"learner_cfg": xxx, "policy": xxx}.\n        Returns:\n            - learner (:obj:`BaseLearner`): Created base learner.\n\n        .. note::\n            Three methods(\'get_data\', \'send_policy\', \'send_learn_info\'), dataloader and policy are set.\n            The reason why they are set here rather than base learner is that, they highly depend on the specific task.\n            Only after task info is passed from coordinator to comm learner through learner slave, can they be\n            clarified and initialized.\n        '
        learner_cfg = EasyDict(task_info['learner_cfg'])
        learner = create_learner(learner_cfg, dist_info=[self._rank, self._world_size], exp_name=learner_cfg.exp_name)
        for item in ['get_data', 'send_policy', 'send_learn_info']:
            setattr(learner, item, getattr(self, item))
        policy_cfg = task_info['policy']
        learner.policy = create_policy(policy_cfg, enable_field=['learn']).learn_mode
        learner.setup_dataloader()
        return learner

def create_comm_learner(cfg: EasyDict) -> BaseCommLearner:
    if False:
        return 10
    "\n    Overview:\n        Given the key(comm_learner_name), create a new comm learner instance if in comm_map's values,\n        or raise an KeyError. In other words, a derived comm learner must first register,\n        then can call ``create_comm_learner`` to get the instance.\n    Arguments:\n        - cfg (:obj:`dict`): Learner config. Necessary keys: [import_names, comm_learner_type].\n    Returns:\n        - learner (:obj:`BaseCommLearner`): The created new comm learner, should be an instance of one of             comm_map's values.\n    "
    import_module(cfg.get('import_names', []))
    return COMM_LEARNER_REGISTRY.build(cfg.type, cfg=cfg)