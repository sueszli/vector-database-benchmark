from abc import ABC, abstractmethod
from typing import Any
from easydict import EasyDict
from ding.utils import get_task_uid, import_module, COMM_COLLECTOR_REGISTRY
from ..base_parallel_collector import create_parallel_collector, BaseParallelCollector

class BaseCommCollector(ABC):
    """
    Overview:
        Abstract baseclass for common collector.
    Interfaces:
        __init__, get_policy_update_info, send_metadata, send_stepdata
        start, close, _create_collector
    Property:
        collector_uid
    """

    def __init__(self, cfg):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Initialization method.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Config dict\n        '
        self._cfg = cfg
        self._end_flag = True
        self._collector_uid = get_task_uid()

    @abstractmethod
    def get_policy_update_info(self, path: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get policy information in corresponding path.\n            Will be registered in base collector.\n        Arguments:\n            - path (:obj:`str`): path to policy update information.\n        '
        raise NotImplementedError

    @abstractmethod
    def send_metadata(self, metadata: Any) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Store meta data in queue, which will be retrieved by callback function "deal_with_collector_data"\n            in collector slave, then will be sent to coordinator.\n            Will be registered in base collector.\n        Arguments:\n            - metadata (:obj:`Any`): meta data.\n        '
        raise NotImplementedError

    @abstractmethod
    def send_stepdata(self, stepdata: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Save step data in corresponding path.\n            Will be registered in base collector.\n        Arguments:\n            - stepdata (:obj:`Any`): step data.\n        '
        raise NotImplementedError

    def start(self) -> None:
        if False:
            return 10
        '\n        Overview:\n            Start comm collector.\n        '
        self._end_flag = False

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Close comm collector.\n        '
        self._end_flag = True

    @property
    def collector_uid(self) -> str:
        if False:
            while True:
                i = 10
        return self._collector_uid

    def _create_collector(self, task_info: dict) -> BaseParallelCollector:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Receive ``task_info`` passed from coordinator and create a collector.\n        Arguments:\n            - task_info (:obj:`dict`): Task info dict from coordinator. Should be like         Returns:\n            - collector (:obj:`BaseParallelCollector`): Created base collector.\n        Note:\n            Four methods('send_metadata', 'send_stepdata', 'get_policy_update_info'), and policy are set.\n            The reason why they are set here rather than base collector is, they highly depend on the specific task.\n            Only after task info is passed from coordinator to comm collector through learner slave, can they be\n            clarified and initialized.\n        "
        collector_cfg = EasyDict(task_info['collector_cfg'])
        collector = create_parallel_collector(collector_cfg)
        for item in ['send_metadata', 'send_stepdata', 'get_policy_update_info']:
            setattr(collector, item, getattr(self, item))
        return collector

def create_comm_collector(cfg: EasyDict) -> BaseCommCollector:
    if False:
        print('Hello World!')
    "\n    Overview:\n        Given the key(comm_collector_name), create a new comm collector instance if in comm_map's values,\n        or raise an KeyError. In other words, a derived comm collector must first register,\n        then can call ``create_comm_collector`` to get the instance.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Collector config. Necessary keys: [import_names, comm_collector_type].\n    Returns:\n        - collector (:obj:`BaseCommCollector`): The created new comm collector, should be an instance of one of         comm_map's values.\n    "
    import_module(cfg.get('import_names', []))
    return COMM_COLLECTOR_REGISTRY.build(cfg.type, cfg=cfg)