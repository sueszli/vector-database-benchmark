from abc import ABC, abstractmethod
from collections import defaultdict
from easydict import EasyDict
import copy
from ding.utils import import_module, COMMANDER_REGISTRY, LimitedSpaceContainer

class BaseCommander(ABC):
    """
    Overview:
        Base parallel commander abstract class.
    Interface:
        get_collector_task
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            while True:
                i = 10
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def get_collector_task(self) -> dict:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def judge_collector_finish(self, task_id: str, info: dict) -> bool:
        if False:
            for i in range(10):
                print('nop')
        collector_done = info.get('collector_done', False)
        if collector_done:
            return True
        return False

    def judge_learner_finish(self, task_id: str, info: dict) -> bool:
        if False:
            while True:
                i = 10
        learner_done = info.get('learner_done', False)
        if learner_done:
            return True
        return False

@COMMANDER_REGISTRY.register('naive')
class NaiveCommander(BaseCommander):
    """
    Overview:
        A naive implementation of parallel commander.
    Interface:
        __init__, get_collector_task, get_learner_task, finsh_collector_task, finish_learner_task,
        notify_fail_collector_task, notify_fail_learner_task, update_learner_info
    """
    config = dict(collector_task_space=1, learner_task_space=1, eval_interval=60)

    def __init__(self, cfg: dict) -> None:
        if False:
            return 10
        '\n        Overview:\n            Init the naive commander according to config\n        Arguments:\n            - cfg (:obj:`dict`): The config to init commander. Should include \\\n                "collector_task_space" and "learner_task_space".\n        '
        self._cfg = cfg
        self._exp_name = cfg.exp_name
        commander_cfg = self._cfg.policy.other.commander
        self._collector_task_space = LimitedSpaceContainer(0, commander_cfg.collector_task_space)
        self._learner_task_space = LimitedSpaceContainer(0, commander_cfg.learner_task_space)
        self._collector_env_cfg = copy.deepcopy(self._cfg.env)
        self._collector_env_cfg.pop('collector_episode_num')
        self._collector_env_cfg.pop('evaluator_episode_num')
        self._collector_env_cfg.manager.episode_num = self._cfg.env.collector_episode_num
        self._collector_task_count = 0
        self._learner_task_count = 0
        self._learner_info = defaultdict(list)
        self._learner_task_finish_count = 0
        self._collector_task_finish_count = 0

    def get_collector_task(self) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Get a new collector task when ``collector_task_count`` is smaller than ``collector_task_space``.\n        Return:\n            - task (:obj:`dict`): New collector task.\n        '
        if self._collector_task_space.acquire_space():
            self._collector_task_count += 1
            collector_cfg = copy.deepcopy(self._cfg.policy.collect.collector)
            collector_cfg.collect_setting = {'eps': 0.9}
            collector_cfg.eval_flag = False
            collector_cfg.policy = copy.deepcopy(self._cfg.policy)
            collector_cfg.policy_update_path = 'test.pth'
            collector_cfg.env = self._collector_env_cfg
            collector_cfg.exp_name = self._exp_name
            return {'task_id': 'collector_task_id{}'.format(self._collector_task_count), 'buffer_id': 'test', 'collector_cfg': collector_cfg}
        else:
            return None

    def get_learner_task(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Get the new learner task when task_count is less than task_space\n        Return:\n            - task (:obj:`dict`): the new learner task\n        '
        if self._learner_task_space.acquire_space():
            self._learner_task_count += 1
            learner_cfg = copy.deepcopy(self._cfg.policy.learn.learner)
            learner_cfg.exp_name = self._exp_name
            return {'task_id': 'learner_task_id{}'.format(self._learner_task_count), 'policy_id': 'test.pth', 'buffer_id': 'test', 'learner_cfg': learner_cfg, 'replay_buffer_cfg': copy.deepcopy(self._cfg.policy.other.replay_buffer), 'policy': copy.deepcopy(self._cfg.policy)}
        else:
            return None

    def finish_collector_task(self, task_id: str, finished_task: dict) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            finish collector task will add the collector_task_finish_count\n        '
        self._collector_task_space.release_space()
        self._collector_task_finish_count += 1

    def finish_learner_task(self, task_id: str, finished_task: dict) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            finish learner task will add the learner_task_finish_count and get the buffer_id of task to close the buffer\n        Return:\n            the finished_task buffer_id\n        '
        self._learner_task_finish_count += 1
        self._learner_task_space.release_space()
        return finished_task['buffer_id']

    def notify_fail_collector_task(self, task: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            naive coordinator will pass when need to notify_fail_collector_task\n        '
        self._collector_task_space.release_space()

    def notify_fail_learner_task(self, task: dict) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            naive coordinator will pass when need to notify_fail_learner_task\n        '
        self._learner_task_space.release_space()

    def update_learner_info(self, task_id: str, info: dict) -> None:
        if False:
            return 10
        '\n        Overview:\n            append the info to learner:\n        Arguments:\n            - task_id (:obj:`str`): the learner task_id\n            - info (:obj:`dict`): the info to append to learner\n        '
        self._learner_info[task_id].append(info)

    def increase_collector_task_space(self):
        if False:
            for i in range(10):
                print('nop')
        '"\n        Overview:\n        Increase task space when a new collector has added dynamically.\n        '
        self._collector_task_space.increase_space()

    def decrease_collector_task_space(self):
        if False:
            while True:
                i = 10
        '"\n        Overview:\n        Decrease task space when a new collector has removed dynamically.\n        '
        self._collector_task_space.decrease_space()

def create_parallel_commander(cfg: EasyDict) -> BaseCommander:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        create the commander according to cfg\n    Arguments:\n        - cfg (:obj:`dict`): the commander cfg to create, should include import_names and parallel_commander_type\n    '
    cfg = EasyDict(cfg)
    import_names = cfg.policy.other.commander.import_names
    import_module(import_names)
    return COMMANDER_REGISTRY.build(cfg.policy.other.commander.type, cfg=cfg)

def get_parallel_commander_cls(cfg: EasyDict) -> type:
    if False:
        return 10
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return COMMANDER_REGISTRY.get(cfg.type)