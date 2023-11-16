from typing import Any, Union, Callable, List, Dict, Optional, Tuple
from ditk import logging
from collections import namedtuple
from functools import partial
from easydict import EasyDict
import copy
from ding.torch_utils import CountVar, auto_checkpoint, build_log_buffer
from ding.utils import build_logger, EasyTimer, import_module, LEARNER_REGISTRY, get_rank, get_world_size
from ding.utils.autolog import LoggedValue, LoggedModel, TickTime
from ding.utils.data import AsyncDataLoader
from .learner_hook import build_learner_hook_by_cfg, add_learner_hook, merge_hooks, LearnerHook

@LEARNER_REGISTRY.register('base')
class BaseLearner(object):
    """
    Overview:
        Base class for policy learning.
    Interface:
        train, call_hook, register_hook, save_checkpoint, start, setup_dataloader, close
    Property:
        learn_info, priority_info, last_iter, train_iter, rank, world_size, policy
        monitor, log_buffer, logger, tb_logger, ckpt_name, exp_name, instance_name
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            return 10
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    config = dict(train_iterations=int(1000000000.0), dataloader=dict(num_workers=0), log_policy=True, hook=dict(load_ckpt_before_run='', log_show_after_iter=100, save_ckpt_after_iter=10000, save_ckpt_after_run=True))
    _name = 'BaseLearner'

    def __init__(self, cfg: EasyDict, policy: namedtuple=None, tb_logger: Optional['SummaryWriter']=None, dist_info: Tuple[int, int]=None, exp_name: Optional[str]='default_experiment', instance_name: Optional[str]='learner') -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialization method, build common learner components according to cfg, such as hook, wrapper and so on.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Learner config, you can refer cls.config for details.\n            - policy (:obj:`namedtuple`): A collection of policy function of learn mode. And policy can also be                 initialized when runtime.\n            - tb_logger (:obj:`SummaryWriter`): Tensorboard summary writer.\n            - dist_info (:obj:`Tuple[int, int]`): Multi-GPU distributed training information.\n            - exp_name (:obj:`str`): Experiment name, which is used to indicate output directory.\n            - instance_name (:obj:`str`): Instance name, which should be unique among different learners.\n        Notes:\n            If you want to debug in sync CUDA mode, please add the following code at the beginning of ``__init__``.\n\n            .. code:: python\n\n                os.environ[\'CUDA_LAUNCH_BLOCKING\'] = "1"  # for debug async CUDA\n        '
        self._cfg = cfg
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._ckpt_name = None
        self._timer = EasyTimer()
        self._end_flag = False
        self._learner_done = False
        if dist_info is None:
            self._rank = get_rank()
            self._world_size = get_world_size()
        else:
            (self._rank, self._world_size) = dist_info
        if self._world_size > 1:
            self._cfg.hook.log_reduce_after_iter = True
        if self._rank == 0:
            if tb_logger is not None:
                (self._logger, _) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False)
                self._tb_logger = tb_logger
            else:
                (self._logger, self._tb_logger) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name)
        else:
            (self._logger, _) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False)
            self._tb_logger = None
        self._log_buffer = {'scalar': build_log_buffer(), 'scalars': build_log_buffer(), 'histogram': build_log_buffer()}
        if policy is not None:
            self.policy = policy
        self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
        self._last_iter = CountVar(init_val=0)
        self._setup_wrapper()
        self._setup_hook()

    def _setup_hook(self) -> None:
        if False:
            return 10
        '\n        Overview:\n            Setup hook for base_learner. Hook is the way to implement some functions at specific time point\n            in base_learner. You can refer to ``learner_hook.py``.\n        '
        if hasattr(self, '_hooks'):
            self._hooks = merge_hooks(self._hooks, build_learner_hook_by_cfg(self._cfg.hook))
        else:
            self._hooks = build_learner_hook_by_cfg(self._cfg.hook)

    def _setup_wrapper(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Use ``_time_wrapper`` to get ``train_time``.\n        Note:\n            ``data_time`` is wrapped in ``setup_dataloader``.\n        '
        self._wrapper_timer = EasyTimer()
        self.train = self._time_wrapper(self.train, 'scalar', 'train_time')

    def _time_wrapper(self, fn: Callable, var_type: str, var_name: str) -> Callable:
        if False:
            return 10
        "\n        Overview:\n            Wrap a function and record the time it used in ``_log_buffer``.\n        Arguments:\n            - fn (:obj:`Callable`): Function to be time_wrapped.\n            - var_type (:obj:`str`): Variable type, e.g. ['scalar', 'scalars', 'histogram'].\n            - var_name (:obj:`str`): Variable name, e.g. ['cur_lr', 'total_loss'].\n        Returns:\n             - wrapper (:obj:`Callable`): The wrapper to acquire a function's time.\n        "

        def wrapper(*args, **kwargs) -> Any:
            if False:
                return 10
            with self._wrapper_timer:
                ret = fn(*args, **kwargs)
            self._log_buffer[var_type][var_name] = self._wrapper_timer.value
            return ret
        return wrapper

    def register_hook(self, hook: LearnerHook) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Add a new learner hook.\n        Arguments:\n            - hook (:obj:`LearnerHook`): The hook to be addedr.\n        '
        add_learner_hook(self._hooks, hook)

    def train(self, data: dict, envstep: int=-1, policy_kwargs: Optional[dict]=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Given training data, implement network update for one iteration and update related variables.\n            Learner's API for serial entry.\n            Also called in ``start`` for each iteration's training.\n        Arguments:\n            - data (:obj:`dict`): Training data which is retrieved from repaly buffer.\n\n        .. note::\n\n            ``_policy`` must be set before calling this method.\n\n            ``_policy.forward`` method contains: forward, backward, grad sync(if in multi-gpu mode) and\n            parameter update.\n\n            ``before_iter`` and ``after_iter`` hooks are called at the beginning and ending.\n        "
        assert hasattr(self, '_policy'), 'please set learner policy'
        self.call_hook('before_iter')
        if policy_kwargs is None:
            policy_kwargs = {}
        log_vars = self._policy.forward(data, **policy_kwargs)
        if isinstance(log_vars, dict):
            priority = log_vars.pop('priority', None)
        elif isinstance(log_vars, list):
            priority = log_vars[-1].pop('priority', None)
        else:
            raise TypeError('not support type for log_vars: {}'.format(type(log_vars)))
        if priority is not None:
            replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data]
            replay_unique_id = [d.get('replay_unique_id', None) for d in data]
            self.priority_info = {'priority': priority, 'replay_buffer_idx': replay_buffer_idx, 'replay_unique_id': replay_unique_id}
        self._collector_envstep = envstep
        if isinstance(log_vars, dict):
            log_vars = [log_vars]
        for elem in log_vars:
            (scalars_vars, histogram_vars) = ({}, {})
            for k in list(elem.keys()):
                if '[scalars]' in k:
                    new_k = k.split(']')[-1]
                    scalars_vars[new_k] = elem.pop(k)
                elif '[histogram]' in k:
                    new_k = k.split(']')[-1]
                    histogram_vars[new_k] = elem.pop(k)
            self._log_buffer['scalar'].update(elem)
            self._log_buffer['scalars'].update(scalars_vars)
            self._log_buffer['histogram'].update(histogram_vars)
            self.call_hook('after_iter')
            self._last_iter.add(1)
        return log_vars

    @auto_checkpoint
    def start(self) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            [Only Used In Parallel Mode] Learner's API for parallel entry.\n            For each iteration, learner will get data through ``_next_data`` and call ``train`` to train.\n\n        .. note::\n\n            ``before_run`` and ``after_run`` hooks are called at the beginning and ending.\n        "
        self._end_flag = False
        self._learner_done = False
        self.call_hook('before_run')
        for i in range(self._cfg.train_iterations):
            data = self._next_data()
            if self._end_flag:
                break
            self.train(data)
        self._learner_done = True
        self.call_hook('after_run')

    def setup_dataloader(self) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            [Only Used In Parallel Mode] Setup learner's dataloader.\n\n        .. note::\n\n            Only in parallel mode will we use attributes ``get_data`` and ``_dataloader`` to get data from file system;\n            Instead, in serial version, we can fetch data from memory directly.\n\n            In parallel mode, ``get_data`` is set by ``LearnerCommHelper``, and should be callable.\n            Users don't need to know the related details if not necessary.\n        "
        cfg = self._cfg.dataloader
        batch_size = self._policy.get_attribute('batch_size')
        device = self._policy.get_attribute('device')
        chunk_size = cfg.chunk_size if 'chunk_size' in cfg else batch_size
        self._dataloader = AsyncDataLoader(self.get_data, batch_size, device, chunk_size, collate_fn=lambda x: x, num_workers=cfg.num_workers)
        self._next_data = self._time_wrapper(self._next_data, 'scalar', 'data_time')

    def _next_data(self) -> Any:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            [Only Used In Parallel Mode] Call ``_dataloader``'s ``__next__`` method to return next training data.\n        Returns:\n            - data (:obj:`Any`): Next training data from dataloader.\n        "
        return next(self._dataloader)

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            [Only Used In Parallel Mode] Close the related resources, e.g. dataloader, tensorboard logger, etc.\n        '
        if self._end_flag:
            return
        self._end_flag = True
        if hasattr(self, '_dataloader'):
            self._dataloader.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        if False:
            while True:
                i = 10
        self.close()

    def call_hook(self, name: str) -> None:
        if False:
            return 10
        "\n        Overview:\n            Call the corresponding hook plugins according to position name.\n        Arguments:\n            - name (:obj:`str`): Hooks in which position to call,                 should be in ['before_run', 'after_run', 'before_iter', 'after_iter'].\n        "
        for hook in self._hooks[name]:
            hook(self)

    def info(self, s: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Log string info by ``self._logger.info``.\n        Arguments:\n            - s (:obj:`str`): The message to add into the logger.\n        '
        self._logger.info('[RANK{}]: {}'.format(self._rank, s))

    def debug(self, s: str) -> None:
        if False:
            i = 10
            return i + 15
        self._logger.debug('[RANK{}]: {}'.format(self._rank, s))

    def save_checkpoint(self, ckpt_name: str=None) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Directly call ``save_ckpt_after_run`` hook to save checkpoint.\n        Note:\n            Must guarantee that "save_ckpt_after_run" is registered in "after_run" hook.\n            This method is called in:\n\n                - ``auto_checkpoint`` (``torch_utils/checkpoint_helper.py``), which is designed for                     saving checkpoint whenever an exception raises.\n                - ``serial_pipeline`` (``entry/serial_entry.py``). Used to save checkpoint when reaching                     new highest episode return.\n        '
        if ckpt_name is not None:
            self.ckpt_name = ckpt_name
        names = [h.name for h in self._hooks['after_run']]
        assert 'save_ckpt_after_run' in names
        idx = names.index('save_ckpt_after_run')
        self._hooks['after_run'][idx](self)
        self.ckpt_name = None

    @property
    def learn_info(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Get current info dict, which will be sent to commander, e.g. replay buffer priority update,\n            current iteration, hyper-parameter adjustment, whether task is finished, etc.\n        Returns:\n            - info (:obj:`dict`): Current learner info dict.\n        '
        ret = {'learner_step': self._last_iter.val, 'priority_info': self.priority_info, 'learner_done': self._learner_done}
        return ret

    @property
    def last_iter(self) -> CountVar:
        if False:
            i = 10
            return i + 15
        return self._last_iter

    @property
    def train_iter(self) -> int:
        if False:
            while True:
                i = 10
        return self._last_iter.val

    @property
    def monitor(self) -> 'TickMonitor':
        if False:
            i = 10
            return i + 15
        return self._monitor

    @property
    def log_buffer(self) -> dict:
        if False:
            print('Hello World!')
        return self._log_buffer

    @log_buffer.setter
    def log_buffer(self, _log_buffer: Dict[str, Dict[str, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        self._log_buffer = _log_buffer

    @property
    def logger(self) -> logging.Logger:
        if False:
            i = 10
            return i + 15
        return self._logger

    @property
    def tb_logger(self) -> 'TensorBoradLogger':
        if False:
            for i in range(10):
                print('nop')
        return self._tb_logger

    @property
    def exp_name(self) -> str:
        if False:
            print('Hello World!')
        return self._exp_name

    @property
    def instance_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._instance_name

    @property
    def rank(self) -> int:
        if False:
            print('Hello World!')
        return self._rank

    @property
    def world_size(self) -> int:
        if False:
            return 10
        return self._world_size

    @property
    def policy(self) -> 'Policy':
        if False:
            for i in range(10):
                print('nop')
        return self._policy

    @policy.setter
    def policy(self, _policy: 'Policy') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Note:\n            Policy variable monitor is set alongside with policy, because variables are determined by specific policy.\n        '
        self._policy = _policy
        if self._rank == 0:
            self._monitor = get_simple_monitor_type(self._policy.monitor_vars())(TickTime(), expire=10)
        if self._cfg.log_policy:
            self.info(self._policy.info())

    @property
    def priority_info(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '_priority_info'):
            self._priority_info = {}
        return self._priority_info

    @priority_info.setter
    def priority_info(self, _priority_info: dict) -> None:
        if False:
            print('Hello World!')
        self._priority_info = _priority_info

    @property
    def ckpt_name(self) -> str:
        if False:
            print('Hello World!')
        return self._ckpt_name

    @ckpt_name.setter
    def ckpt_name(self, _ckpt_name: str) -> None:
        if False:
            print('Hello World!')
        self._ckpt_name = _ckpt_name

def create_learner(cfg: EasyDict, **kwargs) -> BaseLearner:
    if False:
        return 10
    "\n    Overview:\n        Given the key(learner_name), create a new learner instance if in learner_mapping's values,\n        or raise an KeyError. In other words, a derived learner must first register, then can call ``create_learner``\n        to get the instance.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Learner config. Necessary keys: [learner.import_module, learner.learner_type].\n    Returns:\n        - learner (:obj:`BaseLearner`): The created new learner, should be an instance of one of             learner_mapping's values.\n    "
    import_module(cfg.get('import_names', []))
    return LEARNER_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)

class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info during training.
        Info includes: cur_lr, time(data, train, forward, backward), loss(total,...)
        These info variables are firstly recorded in ``log_buffer``, then in ``LearnerHook`` will vars in
        in this monitor be updated by``log_buffer``, finally printed to text logger and tensorboard logger.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    data_time = LoggedValue(float)
    train_time = LoggedValue(float)
    total_collect_step = LoggedValue(float)
    total_step = LoggedValue(float)
    total_episode = LoggedValue(float)
    total_sample = LoggedValue(float)
    total_duration = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):
        if False:
            for i in range(10):
                print('nop')
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):
        if False:
            while True:
                i = 10

        def __avg_func(prop_name: str) -> float:
            if False:
                i = 10
                return i + 15
            records = self.range_values[prop_name]()
            _list = [_value for ((_begin_time, _end_time), _value) in records]
            return sum(_list) / len(_list) if len(_list) != 0 else 0

        def __val_func(prop_name: str) -> float:
            if False:
                print('Hello World!')
            records = self.range_values[prop_name]()
            return records[-1][1]
        for k in getattr(self, '_LoggedModel__properties'):
            self.register_attribute_value('avg', k, partial(__avg_func, prop_name=k))
            self.register_attribute_value('val', k, partial(__val_func, prop_name=k))

def get_simple_monitor_type(properties: List[str]=[]) -> TickMonitor:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Besides basic training variables provided in ``TickMonitor``, many policies have their own customized\n        ones to record and monitor. This function can return a customized tick monitor.\n        Compared with ``TickMonitor``, ``SimpleTickMonitor`` can record extra ``properties`` passed in by a policy.\n    Argumenst:\n         - properties (:obj:`List[str]`): Customized properties to monitor.\n    Returns:\n        - simple_tick_monitor (:obj:`SimpleTickMonitor`): A simple customized tick monitor.\n    '
    if len(properties) == 0:
        return TickMonitor
    else:
        attrs = {}
        properties = ['data_time', 'train_time', 'sample_count', 'total_collect_step', 'total_step', 'total_sample', 'total_episode', 'total_duration'] + properties
        for p_name in properties:
            attrs[p_name] = LoggedValue(float)
        return type('SimpleTickMonitor', (TickMonitor,), attrs)