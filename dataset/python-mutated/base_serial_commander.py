from collections import namedtuple
from easydict import EasyDict
import copy

class BaseSerialCommander(object):
    """
    Overview:
        Base serial commander class.
    Interface:
        __init__, step
    Property:
        policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            while True:
                i = 10
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    config = {}

    def __init__(self, cfg: dict, learner: 'BaseLearner', collector: 'BaseSerialCollector', evaluator: 'InteractionSerialEvaluator', replay_buffer: 'IBuffer', policy: namedtuple=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Init the BaseSerialCommander\n        Arguments:\n            - cfg (:obj:`dict`): the config of commander\n            - learner (:obj:`BaseLearner`): the learner\n            - collector (:obj:`BaseSerialCollector`): the collector\n            - evaluator (:obj:`InteractionSerialEvaluator`): the evaluator\n            - replay_buffer (:obj:`IBuffer`): the buffer\n        '
        self._cfg = cfg
        self._learner = learner
        self._collector = collector
        self._evaluator = evaluator
        self._replay_buffer = replay_buffer
        self._info = {}
        if policy is not None:
            self.policy = policy

    def step(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Step the commander\n        '
        learn_info = self._learner.learn_info
        collector_info = {'envstep': self._collector.envstep}
        self._info.update(learn_info)
        self._info.update(collector_info)
        collect_kwargs = self._policy.get_setting_collect(self._info)
        return collect_kwargs

    @property
    def policy(self) -> 'Policy':
        if False:
            print('Hello World!')
        return self._policy

    @policy.setter
    def policy(self, _policy: 'Policy') -> None:
        if False:
            print('Hello World!')
        self._policy = _policy