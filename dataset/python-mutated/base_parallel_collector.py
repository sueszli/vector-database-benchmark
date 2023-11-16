from typing import Any, Union, Tuple
from abc import ABC, abstractmethod
import sys
from ditk import logging
import copy
from collections import namedtuple
from functools import partial
from easydict import EasyDict
import torch
from ding.policy import Policy
from ding.envs import BaseEnvManager
from ding.utils.autolog import LoggedValue, LoggedModel, TickTime
from ding.utils import build_logger, EasyTimer, get_task_uid, import_module, pretty_print, PARALLEL_COLLECTOR_REGISTRY
from ding.torch_utils import build_log_buffer, to_tensor, to_ndarray

class BaseParallelCollector(ABC):
    """
    Overview:
        Abstract baseclass for collector.
    Interfaces:
        __init__, info, error, debug, get_finish_info, start, close, _setup_timer, _setup_logger, _iter_after_hook,
        _policy_inference, _env_step, _process_timestep, _finish_task, _update_policy, _start_thread, _join_thread
    Property:
        policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            return 10
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Initialization method.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Config dict\n        '
        self._cfg = cfg
        self._eval_flag = cfg.eval_flag
        self._prefix = 'EVALUATOR' if self._eval_flag else 'COLLECTOR'
        self._collector_uid = get_task_uid()
        (self._logger, self._monitor, self._log_buffer) = self._setup_logger()
        self._end_flag = False
        self._setup_timer()
        self._iter_count = 0
        self.info('\nCFG INFO:\n{}'.format(pretty_print(cfg, direct_print=False)))

    def info(self, s: str) -> None:
        if False:
            print('Hello World!')
        self._logger.info('[{}({})]: {}'.format(self._prefix, self._collector_uid, s))

    def debug(self, s: str) -> None:
        if False:
            return 10
        self._logger.debug('[{}({})]: {}'.format(self._prefix, self._collector_uid, s))

    def error(self, s: str) -> None:
        if False:
            return 10
        self._logger.error('[{}({})]: {}'.format(self._prefix, self._collector_uid, s))

    def _setup_timer(self) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Setup TimeWrapper for base_collector. TimeWrapper is a decent timer wrapper that can be used easily.\n            You can refer to ``ding/utils/time_helper.py``.\n\n        Note:\n            - _policy_inference (:obj:`Callable`): The wrapper to acquire a policy's time.\n            - _env_step (:obj:`Callable`): The wrapper to acquire a environment's time.\n        "
        self._timer = EasyTimer()

        def policy_wrapper(fn):
            if False:
                return 10

            def wrapper(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                with self._timer:
                    ret = fn(*args, **kwargs)
                self._log_buffer['policy_time'] = self._timer.value
                return ret
            return wrapper

        def env_wrapper(fn):
            if False:
                for i in range(10):
                    print('nop')

            def wrapper(*args, **kwargs):
                if False:
                    return 10
                with self._timer:
                    ret = fn(*args, **kwargs)
                size = sys.getsizeof(ret) / (1024 * 1024)
                self._log_buffer['env_time'] = self._timer.value
                self._log_buffer['timestep_size'] = size
                self._log_buffer['norm_env_time'] = self._timer.value / size
                return ret
            return wrapper
        self._policy_inference = policy_wrapper(self._policy_inference)
        self._env_step = env_wrapper(self._env_step)

    def _setup_logger(self) -> Tuple[logging.Logger, 'TickMonitor', 'LogDict']:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Setup logger for base_collector. Logger includes logger, monitor and log buffer dict.\n        Returns:\n            - logger (:obj:`logging.Logger`): logger that displays terminal output\n            - monitor (:obj:`TickMonitor`): monitor that is related info of one interation with env\n            - log_buffer (:obj:`LogDict`): log buffer dict\n        '
        path = './{}/log/{}'.format(self._cfg.exp_name, self._prefix.lower())
        name = '{}'.format(self._collector_uid)
        (logger, _) = build_logger(path, name, need_tb=False)
        monitor = TickMonitor(TickTime(), expire=self._cfg.print_freq * 2)
        log_buffer = build_log_buffer()
        return (logger, monitor, log_buffer)

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        self._end_flag = False
        self._update_policy()
        self._start_thread()
        while not self._end_flag:
            obs = self._env_manager.ready_obs
            obs = to_tensor(obs, dtype=torch.float32)
            action = self._policy_inference(obs)
            action = to_ndarray(action)
            timestep = self._env_step(action)
            timestep = to_tensor(timestep, dtype=torch.float32)
            self._process_timestep(timestep)
            self._iter_after_hook()
            if self._env_manager.done:
                break

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._end_flag:
            return
        self._end_flag = True
        self._join_thread()

    def _iter_after_hook(self):
        if False:
            i = 10
            return i + 15
        for (k, v) in self._log_buffer.items():
            setattr(self._monitor, k, v)
        self._monitor.time.step()
        if self._iter_count % self._cfg.print_freq == 0:
            self.debug('{}TimeStep{}{}'.format('=' * 35, self._iter_count, '=' * 35))
            var_dict = {}
            for k in self._log_buffer:
                for attr in self._monitor.get_property_attribute(k):
                    k_attr = k + '_' + attr
                    var_dict[k_attr] = getattr(self._monitor, attr)[k]()
            self._logger.debug(self._logger.get_tabulate_vars_hor(var_dict))
        self._log_buffer.clear()
        self._iter_count += 1

    @abstractmethod
    def get_finish_info(self) -> dict:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def _policy_inference(self, obs: Any) -> Any:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @abstractmethod
    def _env_step(self, action: Any) -> Any:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def _process_timestep(self, timestep: namedtuple) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def _update_policy(self) -> None:
        if False:
            return 10
        raise NotImplementedError

    def _start_thread(self) -> None:
        if False:
            return 10
        pass

    def _join_thread(self) -> None:
        if False:
            return 10
        pass

    @property
    def policy(self) -> Policy:
        if False:
            for i in range(10):
                print('nop')
        return self._policy

    @policy.setter
    def policy(self, _policy: Policy) -> None:
        if False:
            print('Hello World!')
        self._policy = _policy

    @property
    def env_manager(self) -> BaseEnvManager:
        if False:
            i = 10
            return i + 15
        return self._env_manager

    @env_manager.setter
    def env_manager(self, _env_manager: BaseEnvManager) -> None:
        if False:
            while True:
                i = 10
        self._env_manager = _env_manager

def create_parallel_collector(cfg: EasyDict) -> BaseParallelCollector:
    if False:
        return 10
    import_module(cfg.get('import_names', []))
    return PARALLEL_COLLECTOR_REGISTRY.build(cfg.type, cfg=cfg)

def get_parallel_collector_cls(cfg: EasyDict) -> type:
    if False:
        for i in range(10):
            print('nop')
    import_module(cfg.get('import_names', []))
    return PARALLEL_COLLECTOR_REGISTRY.get(cfg.type)

class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info of one interation with env.
        Info include: policy_time, env_time, norm_env_time, timestep_size...
        These info variables would first be recorded in ``log_buffer``, then in ``self._iter_after_hook`` will vars in
        in this monitor be updated by``log_buffer``, then printed to text logger and tensorboard logger.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    policy_time = LoggedValue(float)
    env_time = LoggedValue(float)
    timestep_size = LoggedValue(float)
    norm_env_time = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):
        if False:
            print('Hello World!')
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):
        if False:
            for i in range(10):
                print('nop')

        def __avg_func(prop_name: str) -> float:
            if False:
                i = 10
                return i + 15
            records = self.range_values[prop_name]()
            _list = [_value for ((_begin_time, _end_time), _value) in records]
            return sum(_list) / len(_list) if len(_list) != 0 else 0
        self.register_attribute_value('avg', 'policy_time', partial(__avg_func, prop_name='policy_time'))
        self.register_attribute_value('avg', 'env_time', partial(__avg_func, prop_name='env_time'))
        self.register_attribute_value('avg', 'timestep_size', partial(__avg_func, prop_name='timestep_size'))
        self.register_attribute_value('avg', 'norm_env_time', partial(__avg_func, prop_name='norm_env_time'))