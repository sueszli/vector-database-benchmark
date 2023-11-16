import numbers
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from easydict import EasyDict
import ding
from ding.utils import allreduce, read_file, save_file, get_rank

class Hook(ABC):
    """
    Overview:
        Abstract class for hooks.
    Interfaces:
        __init__, __call__
    Property:
        name, priority
    """

    def __init__(self, name: str, priority: float, **kwargs) -> None:
        if False:
            return 10
        "\n        Overview:\n            Init method for hooks. Set name and priority.\n        Arguments:\n            - name (:obj:`str`): The name of hook\n            - priority (:obj:`float`): The priority used in ``call_hook``'s calling sequence.                 Lower value means higher priority.\n        "
        self._name = name
        assert priority >= 0, 'invalid priority value: {}'.format(priority)
        self._priority = priority

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._name

    @property
    def priority(self) -> float:
        if False:
            print('Hello World!')
        return self._priority

    @abstractmethod
    def __call__(self, engine: Any) -> Any:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Should be overwritten by subclass.\n        Arguments:\n            - engine (:obj:`Any`): For LearnerHook, it should be ``BaseLearner`` or its subclass.\n        '
        raise NotImplementedError

class LearnerHook(Hook):
    """
    Overview:
        Abstract class for hooks used in Learner.
    Interfaces:
        __init__
    Property:
        name, priority, position

    .. note::

        Subclass should implement ``self.__call__``.
    """
    positions = ['before_run', 'after_run', 'before_iter', 'after_iter']

    def __init__(self, *args, position: str, **kwargs) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Init LearnerHook.\n        Arguments:\n            - position (:obj:`str`): The position to call hook in learner.                 Must be in ['before_run', 'after_run', 'before_iter', 'after_iter'].\n        "
        super().__init__(*args, **kwargs)
        assert position in self.positions
        self._position = position

    @property
    def position(self) -> str:
        if False:
            print('Hello World!')
        return self._position

class LoadCkptHook(LearnerHook):
    """
    Overview:
        Hook to load checkpoint
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict=EasyDict(), **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Init LoadCkptHook.\n        Arguments:\n            - ext_args (:obj:`EasyDict`): Extended arguments. Use ``ext_args.freq`` to set ``load_ckpt_freq``.\n        '
        super().__init__(*args, **kwargs)
        self._load_path = ext_args['load_path']

    def __call__(self, engine: 'BaseLearner') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Load checkpoint to learner. Checkpoint info includes policy state_dict and iter num.\n        Arguments:\n            - engine (:obj:`BaseLearner`): The BaseLearner to load checkpoint to.\n        '
        path = self._load_path
        if path == '':
            return
        state_dict = read_file(path)
        if 'last_iter' in state_dict:
            last_iter = state_dict.pop('last_iter')
            engine.last_iter.update(last_iter)
        engine.policy.load_state_dict(state_dict)
        engine.info('{} load ckpt in {}'.format(engine.instance_name, path))

class SaveCkptHook(LearnerHook):
    """
    Overview:
        Hook to save checkpoint
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict=EasyDict(), **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            init SaveCkptHook\n        Arguments:\n            - ext_args (:obj:`EasyDict`): extended_args, use ext_args.freq to set save_ckpt_freq\n        '
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:
        if False:
            return 10
        '\n        Overview:\n            Save checkpoint in corresponding path.\n            Checkpoint info includes policy state_dict and iter num.\n        Arguments:\n            - engine (:obj:`BaseLearner`): the BaseLearner which needs to save checkpoint\n        '
        if engine.rank == 0 and engine.last_iter.val % self._freq == 0:
            if engine.instance_name == 'learner':
                dirname = './{}/ckpt'.format(engine.exp_name)
            else:
                dirname = './{}/ckpt_{}'.format(engine.exp_name, engine.instance_name)
            if not os.path.exists(dirname):
                try:
                    os.makedirs(dirname)
                except FileExistsError:
                    pass
            ckpt_name = engine.ckpt_name if engine.ckpt_name else 'iteration_{}.pth.tar'.format(engine.last_iter.val)
            path = os.path.join(dirname, ckpt_name)
            state_dict = engine.policy.state_dict()
            state_dict.update({'last_iter': engine.last_iter.val})
            save_file(path, state_dict)
            engine.info('{} save ckpt in {}'.format(engine.instance_name, path))

class LogShowHook(LearnerHook):
    """
    Overview:
        Hook to show log
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict=EasyDict(), **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            init LogShowHook\n        Arguments:\n            - ext_args (:obj:`EasyDict`): extended_args, use ext_args.freq to set freq\n        '
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Show log, update record and tb_logger if rank is 0 and at interval iterations,\n            clear the log buffer for all learners regardless of rank\n        Arguments:\n            - engine (:obj:`BaseLearner`): the BaseLearner\n        '
        if engine.rank != 0:
            for k in engine.log_buffer:
                engine.log_buffer[k].clear()
            return
        for (k, v) in engine.log_buffer['scalar'].items():
            setattr(engine.monitor, k, v)
        engine.monitor.time.step()
        iters = engine.last_iter.val
        if iters % self._freq == 0:
            engine.info('=== Training Iteration {} Result ==='.format(iters))
            var_dict = {}
            log_vars = engine.policy.monitor_vars()
            attr = 'avg'
            for k in log_vars:
                k_attr = k + '_' + attr
                var_dict[k_attr] = getattr(engine.monitor, attr)[k]()
            engine.logger.info(engine.logger.get_tabulate_vars_hor(var_dict))
            for (k, v) in var_dict.items():
                engine.tb_logger.add_scalar('{}_iter/'.format(engine.instance_name) + k, v, iters)
                engine.tb_logger.add_scalar('{}_step/'.format(engine.instance_name) + k, v, engine._collector_envstep)
            tb_var_dict = {}
            for k in engine.log_buffer['histogram']:
                new_k = '{}/'.format(engine.instance_name) + k
                tb_var_dict[new_k] = engine.log_buffer['histogram'][k]
            for (k, v) in tb_var_dict.items():
                engine.tb_logger.add_histogram(k, v, iters)
        for k in engine.log_buffer:
            engine.log_buffer[k].clear()

class LogReduceHook(LearnerHook):
    """
    Overview:
        Hook to reduce the distributed(multi-gpu) logs
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict=EasyDict(), **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            init LogReduceHook\n        Arguments:\n            - ext_args (:obj:`EasyDict`): extended_args, use ext_args.freq to set log_reduce_freq\n        '
        super().__init__(*args, **kwargs)

    def __call__(self, engine: 'BaseLearner') -> None:
        if False:
            return 10
        '\n        Overview:\n            reduce the logs from distributed(multi-gpu) learners\n        Arguments:\n            - engine (:obj:`BaseLearner`): the BaseLearner\n        '

        def aggregate(data):
            if False:
                print('Hello World!')
            '\n            Overview:\n                aggregate the information from all ranks(usually use sync allreduce)\n            Arguments:\n                - data (:obj:`dict`): Data that needs to be reduced. \\\n                    Could be dict, torch.Tensor, numbers.Integral or numbers.Real.\n            Returns:\n                - new_data (:obj:`dict`): data after reduce\n            '
            if isinstance(data, dict):
                new_data = {k: aggregate(v) for (k, v) in data.items()}
            elif isinstance(data, list) or isinstance(data, tuple):
                new_data = [aggregate(t) for t in data]
            elif isinstance(data, torch.Tensor):
                new_data = data.clone().detach()
                if ding.enable_linklink:
                    allreduce(new_data)
                else:
                    new_data = new_data.to(get_rank())
                    allreduce(new_data)
                    new_data = new_data.cpu()
            elif isinstance(data, numbers.Integral) or isinstance(data, numbers.Real):
                new_data = torch.scalar_tensor(data).reshape([1])
                if ding.enable_linklink:
                    allreduce(new_data)
                else:
                    new_data = new_data.to(get_rank())
                    allreduce(new_data)
                    new_data = new_data.cpu()
                new_data = new_data.item()
            else:
                raise TypeError('invalid type in reduce: {}'.format(type(data)))
            return new_data
        engine.log_buffer = aggregate(engine.log_buffer)
hook_mapping = {'load_ckpt': LoadCkptHook, 'save_ckpt': SaveCkptHook, 'log_show': LogShowHook, 'log_reduce': LogReduceHook}

def register_learner_hook(name: str, hook_type: type) -> None:
    if False:
        print('Hello World!')
    "\n    Overview:\n        Add a new LearnerHook class to hook_mapping, so you can build one instance with `build_learner_hook_by_cfg`.\n    Arguments:\n        - name (:obj:`str`): name of the register hook\n        - hook_type (:obj:`type`): the register hook_type you implemented that realize LearnerHook\n    Examples:\n        >>> class HookToRegister(LearnerHook):\n        >>>     def __init__(*args, **kargs):\n        >>>         ...\n        >>>         ...\n        >>>     def __call__(*args, **kargs):\n        >>>         ...\n        >>>         ...\n        >>> ...\n        >>> register_learner_hook('name_of_hook', HookToRegister)\n        >>> ...\n        >>> hooks = build_learner_hook_by_cfg(cfg)\n    "
    assert issubclass(hook_type, LearnerHook)
    hook_mapping[name] = hook_type
simplified_hook_mapping = {'log_show_after_iter': lambda freq: hook_mapping['log_show']('log_show', 20, position='after_iter', ext_args=EasyDict({'freq': freq})), 'load_ckpt_before_run': lambda path: hook_mapping['load_ckpt']('load_ckpt', 20, position='before_run', ext_args=EasyDict({'load_path': path})), 'save_ckpt_after_iter': lambda freq: hook_mapping['save_ckpt']('save_ckpt_after_iter', 20, position='after_iter', ext_args=EasyDict({'freq': freq})), 'save_ckpt_after_run': lambda _: hook_mapping['save_ckpt']('save_ckpt_after_run', 20, position='after_run'), 'log_reduce_after_iter': lambda _: hook_mapping['log_reduce']('log_reduce_after_iter', 10, position='after_iter')}

def find_char(s: str, flag: str, num: int, reverse: bool=False) -> int:
    if False:
        print('Hello World!')
    assert num > 0, num
    count = 0
    iterable_obj = reversed(range(len(s))) if reverse else range(len(s))
    for i in iterable_obj:
        if s[i] == flag:
            count += 1
            if count == num:
                return i
    return -1

def build_learner_hook_by_cfg(cfg: EasyDict) -> Dict[str, List[Hook]]:
    if False:
        return 10
    "\n    Overview:\n        Build the learner hooks in hook_mapping by config.\n        This function is often used to initialize ``hooks`` according to cfg,\n        while add_learner_hook() is often used to add an existing LearnerHook to `hooks`.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Config dict. Should be like {'hook': xxx}.\n    Returns:\n        - hooks (:obj:`Dict[str, List[Hook]`): Keys should be in ['before_run', 'after_run', 'before_iter',             'after_iter'], each value should be a list containing all hooks in this position.\n    Note:\n        Lower value means higher priority.\n    "
    hooks = {k: [] for k in LearnerHook.positions}
    for (key, value) in cfg.items():
        if key in simplified_hook_mapping and (not isinstance(value, dict)):
            pos = key[find_char(key, '_', 2, reverse=True) + 1:]
            hook = simplified_hook_mapping[key](value)
            priority = hook.priority
        else:
            priority = value.get('priority', 100)
            pos = value.position
            ext_args = value.get('ext_args', {})
            hook = hook_mapping[value.type](value.name, priority, position=pos, ext_args=ext_args)
        idx = 0
        for i in reversed(range(len(hooks[pos]))):
            if priority >= hooks[pos][i].priority:
                idx = i + 1
                break
        hooks[pos].insert(idx, hook)
    return hooks

def add_learner_hook(hooks: Dict[str, List[Hook]], hook: LearnerHook) -> None:
    if False:
        print('Hello World!')
    "\n    Overview:\n        Add a learner hook(:obj:`LearnerHook`) to hooks(:obj:`Dict[str, List[Hook]`)\n    Arguments:\n        - hooks (:obj:`Dict[str, List[Hook]`): You can refer to ``build_learner_hook_by_cfg``'s return ``hooks``.\n        - hook (:obj:`LearnerHook`): The LearnerHook which will be added to ``hooks``.\n    "
    position = hook.position
    priority = hook.priority
    idx = 0
    for i in reversed(range(len(hooks[position]))):
        if priority >= hooks[position][i].priority:
            idx = i + 1
            break
    assert isinstance(hook, LearnerHook)
    hooks[position].insert(idx, hook)

def merge_hooks(hooks1: Dict[str, List[Hook]], hooks2: Dict[str, List[Hook]]) -> Dict[str, List[Hook]]:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Merge two hooks dict, which have the same keys, and each value is sorted by hook priority with stable method.\n    Arguments:\n        - hooks1 (:obj:`Dict[str, List[Hook]`): hooks1 to be merged.\n        - hooks2 (:obj:`Dict[str, List[Hook]`): hooks2 to be merged.\n    Returns:\n        - new_hooks (:obj:`Dict[str, List[Hook]`): New merged hooks dict.\n    Note:\n        This merge function uses stable sort method without disturbing the same priority hook.\n    '
    assert set(hooks1.keys()) == set(hooks2.keys())
    new_hooks = {}
    for k in hooks1.keys():
        new_hooks[k] = sorted(hooks1[k] + hooks2[k], key=lambda x: x.priority)
    return new_hooks

def show_hooks(hooks: Dict[str, List[Hook]]) -> None:
    if False:
        i = 10
        return i + 15
    for k in hooks.keys():
        print('{}: {}'.format(k, [x.__class__.__name__ for x in hooks[k]]))