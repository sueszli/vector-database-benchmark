from typing import Any, Union, List, Optional, Mapping
from logging import warning
from functools import partial, wraps
from abc import abstractmethod
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteModule, _LiteOptimizer
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from bigdl.nano.utils.common import _avx512_checker
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_11, TORCH_VERSION_LESS_1_13, check_ccl
from bigdl.nano.deps.ipex.ipex_api import ipex_optimize
from bigdl.nano.pytorch.strategies import IPEXStrategy, DDPSpawnStrategy, DDPSubprocessStrategy, create_ray_strategy, DDPK8sStrategy

class _TorchNanoModule(_LiteModule):

    def __init__(self, module, precision_plugin, channels_last) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(module, precision_plugin)
        self.channels_last = channels_last

    def state_dict(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(self.module, DistributedDataParallel):
            return self.module.module.state_dict(*args, **kwargs)
        else:
            return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool=True):
        if False:
            while True:
                i = 10
        invalidInputError(TORCH_VERSION_LESS_1_13, "TorchNano doesn't support loading state dict with PyTorch<1.13, please load it using original pytorch model")
        if isinstance(self.module, DistributedDataParallel):
            return self.module.module.load_state_dict(state_dict=state_dict, strict=strict)
        else:
            return self.module.load_state_dict(state_dict=state_dict, strict=strict)

    def __getattr__(self, name: str):
        if False:
            return 10
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if isinstance(self.module, DistributedDataParallel):
            try:
                return getattr(self.module, name)
            except AttributeError:
                pass
            return getattr(self.module.module, name)
        else:
            return getattr(self.module, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        'Casts all inputs to the right memory format.'
        if self.channels_last:

            def _convert_to_channels_last(t: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                if t.dim() == 4:
                    return t.to(memory_format=torch.channels_last)
                return t
            (args, kwargs) = apply_to_collection([args, kwargs], function=_convert_to_channels_last, dtype=torch.Tensor)
        return super().forward(*args, **kwargs)

class _TorchNanoOptimizer(_LiteOptimizer):

    def __init__(self, optimizer: Optimizer, strategy: Strategy, auto_lr: bool, num_processes: Optional[int]) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(optimizer, strategy)
        self.cur_lr_ratio = 1.0
        self.max_lr_ratio = num_processes
        self.cur_step = 0
        self.max_step = 1000
        self.auto_lr = auto_lr

    def step(self, closure=None) -> Any:
        if False:
            i = 10
            return i + 15
        if not self.auto_lr or self.max_lr_ratio is None or self.max_lr_ratio == 1:
            return super().step(closure)
        else:
            base_lrs = []
            for param_group in self.optimizer.param_groups:
                base_lr = param_group['lr']
                base_lrs.append(base_lr)
                param_group['lr'] = base_lr * self.cur_lr_ratio
            ret = super().step(closure=closure)
            for (param_group, base_lr) in zip(self.optimizer.param_groups, base_lrs):
                param_group['lr'] = base_lr
            if self.cur_step < self.max_step:
                self.cur_step += 1
                self.cur_lr_ratio = (self.max_lr_ratio - 1) * self.cur_step / self.max_step + 1
            return ret
distributed_backends = ['spawn', 'ray', 'subprocess', 'k8s']
backends_class_map = {'spawn': DDPSpawnStrategy, 'subprocess': DDPSubprocessStrategy, 'ray': create_ray_strategy, 'k8s': DDPK8sStrategy}

class TorchNano(LightningLite):
    """
    TorchNano for BigDL-Nano pytorch.

    It can be used to accelerate custom pytorch training loops with very few code changes.
    """

    def __init__(self, num_processes: Optional[int]=None, use_ipex: bool=False, distributed_backend: str='subprocess', process_group_backend: Optional[str]=None, precision: Union[str, int]=32, cpu_for_each_process: Optional[List[List[int]]]=None, channels_last: bool=False, auto_lr: bool=True, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Create a TorchNano with nano acceleration.\n\n        :param num_processes: number of processes in distributed training, defaults to ``1``\n        :param use_ipex: whether use ipex acceleration, defaults to ``False``\n        :param distributed_backend: use which backend in distributed mode, defaults to\n            ``'subprocess'``, now avaiable backends are ``'spawn'``, ``'subprocess'`` and ``'ray'``\n        :param process_group_backend: use which process group backend in distributed mode, defaults\n            to ``None``, means using ``'gloo'`` with CPU, while using ``'nccl'`` with GPU, now\n            avaiable backends are ``None`` and ``'ccl'``.\n        :param precision: Double precision (``64``), full precision (``32``),\n            half precision (``16``) or bfloat16 precision (``'bf16'``), defaults to ``32``.\n            Enable ipex bfloat16 weight prepack when ``use_ipex=True`` and ``precision='bf16'``\n        :param cpu_for_each_process: specify the cpu cores which will be used by each process,\n            if ``None``, cpu cores will be distributed evenly by all processes,\n            only take effect when ``num_processes`` > 1\n        :param channels_last: whether convert input to channels last memory formats,\n            defaults to ``False``.\n        :param auto_lr: whether to scale the learning rate linearly by ``num_processes`` times.\n            Defaults to ``True``.\n            If ``num_processes=1`` or other ``lr_scheduler`` is set, ``auto_lr`` will be ignored.\n        "
        self.num_processes = num_processes
        self.use_ipex = use_ipex
        self.dtype = None
        self.cpu_for_each_process = cpu_for_each_process
        self.channels_last = channels_last
        self.auto_lr = auto_lr
        if self.use_ipex and precision == 'bf16':
            self.dtype = torch.bfloat16
            precision = 32
        if self.use_ipex and (not _avx512_checker()):
            if TORCH_VERSION_LESS_1_11:
                warning('Enable ipex<=1.10 in a cpu instruction set without avx512 will crash.Fall back to regular pytorch.')
                self.use_ipex = False
            elif self.dtype == torch.bfloat16:
                warning('Enable IPEX bfloat16 in a cpu instruction set without avx512 will crash. Using 32-bit precision')
                self.dtype = None
        kwargs['precision'] = precision
        if self.num_processes is None and distributed_backend != 'k8s':
            self.num_processes = 1
        if self.num_processes == 1:
            if self.use_ipex:
                strategy = IPEXStrategy(dtype=self.dtype)
            else:
                strategy = None
        elif distributed_backend in backends_class_map:
            check_ccl(process_group_backend)
            cls = backends_class_map[distributed_backend]
            strategy = cls(num_processes=self.num_processes, cpu_for_each_process=self.cpu_for_each_process, use_ipex=self.use_ipex, dtype=self.dtype, process_group_backend=process_group_backend)
        else:
            warning(f"BigDL-Nano doesn't support '{distributed_backend}' backend now, '{distributed_backend}' strategy of pytorch_lightning will be used. Supported backends are 'spawn', 'subprocess' and 'ray'.")
            strategy = distributed_backend
        kwargs['strategy'] = strategy
        super().__init__(*args, **kwargs)
        setattr(self, 'train', partial(self._run_impl, self.train))

    def _setup(self, model: nn.Module, optimizers: List[Optimizer], move_to_device: bool=True) -> Any:
        if False:
            for i in range(10):
                print('nop')
        "Used to replace LightningLite's setup method."
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        self._validate_setup(model, optimizers)
        if move_to_device:
            model = self._move_model_to_device(model=model, optimizers=optimizers)
        (model, optimizers) = self._strategy._setup_model_and_optimizers(model, optimizers)
        if self.use_ipex:
            ret = ipex_optimize(model, optimizers=optimizers, inplace=True, dtype=self.dtype)
            if isinstance(ret, tuple):
                (model, optimizers) = (ret[0], [ret[1]])
            else:
                model = ret
        model = _TorchNanoModule(model, self._precision_plugin, self.channels_last)
        optimizers = [_TorchNanoOptimizer(optimizer, self._strategy, self.auto_lr, self.num_processes) for optimizer in optimizers]
        self._models_setup += 1
        return (model, optimizers)

    def setup(self, model: nn.Module, optimizer: Union[Optimizer, List[Optimizer]], *dataloaders: DataLoader, move_to_device: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setup model, optimizers and dataloaders for accelerated training.\n\n        :param model: A model to setup\n        :param optimizer: The optimizer(s) to setup\n        :param *dataloaders: The dataloader(s) to setup\n        :param move_to_device: If set ``True`` (default), moves the model to the correct device.\n            Set this to ``False`` and alternatively use :meth:`to_device` manually.\n        :return: The tuple of the wrapped model, optimizer, loss_func and dataloaders,\n            in the same order they were passed in.\n        '
        optimizers = [optimizer] if isinstance(optimizer, Optimizer) else optimizer
        (model, optimizers) = self._setup(model, optimizers, move_to_device=move_to_device)
        dataloaders = self.setup_dataloaders(*dataloaders, move_to_device=move_to_device)
        optimizer = optimizers[0] if isinstance(optimizer, Optimizer) else optimizers
        if len(dataloaders) == 0:
            return (model, optimizer)
        else:
            return (model, optimizer, dataloaders)

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            return 10
        '\n        All the code inside this train method gets accelerated by TorchNano.\n\n        You can pass arbitrary arguments to this function when overriding it.\n        '

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        "Only for compatibility, don't use it."
        pass

def _search_setup_args(_models, _optimizers, _dataloaders, args):
    if False:
        i = 10
        return i + 15
    for (idx, value) in enumerate(args):
        if isinstance(value, DataLoader):
            _dataloaders.append((value, args, idx))
        if isinstance(value, nn.Module) and (not isinstance(value, torch.nn.modules.loss._Loss)):
            _models.append((value, args, idx))
        if isinstance(value, Optimizer):
            _optimizers.append((value, args, idx))

def _update_args(objs, obj_pos):
    if False:
        while True:
            i = 10
    for (obj, pos) in zip(objs, obj_pos):
        (_, arg, idx) = pos
        arg[idx] = obj

class _DecoratedTorchNano(TorchNano):

    def train(self, func, *inner_args, **inner_kwargs):
        if False:
            print('Hello World!')
        _model_pos = []
        _optimizer_pos = []
        _data_loader_pos = []
        _inner_args = list(inner_args)
        _search_setup_args(_model_pos, _optimizer_pos, _data_loader_pos, _inner_args)
        _search_setup_args(_model_pos, _optimizer_pos, _data_loader_pos, inner_kwargs)
        invalidInputError(len(_model_pos) == 1, f'there should be only one nn.Module in the function parameter list, but got {len(_model_pos)}')
        _model = _model_pos[0][0]
        _optimizers = [opt[0] for opt in _optimizer_pos]
        _dataloaders = [opt[0] for opt in _data_loader_pos]
        (_setup_model, _setup_optimizers) = self.setup(_model, _optimizers)
        _setup_dataloaders = self.setup_dataloaders(*_dataloaders)
        if len(_dataloaders) == 1:
            _setup_dataloaders = [_setup_dataloaders]
        _update_args([_setup_model], _model_pos)
        _update_args(_setup_optimizers, _optimizer_pos)
        _update_args(_setup_dataloaders, _data_loader_pos)
        return func(*_inner_args, **inner_kwargs)

def nano(num_processes: Optional[int]=None, use_ipex: bool=False, distributed_backend: str='subprocess', precision: Union[str, int]=32, cpu_for_each_process: Optional[List[List[int]]]=None, channels_last: bool=False, auto_lr: bool=True, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Run ``TorchNano.train`` through a convenient decorator function.\n\n    :param num_processes: number of processes in distributed training, defaults to ``1``\n    :param use_ipex: whether use ipex acceleration, defaults to ``False``\n    :param distributed_backend: use which backend in distributed mode, defaults to\n        ``'subprocess'``, now avaiable backends are ``'subprocess'`` and ``'ray'``.\n        ``bigdl.nano.pytorch.nano`` decorator does not support ``'spawn'``.\n    :param precision: Double precision (``64``), full precision (``32``), half precision (``16``)\n        or bfloat16 precision (``'bf16'``), defaults to ``32``.\n        Enable ipex bfloat16 weight prepack when ``use_ipex=True`` and ``precision='bf16'``\n    :param cpu_for_each_process: specify the cpu cores which will be used by each process,\n        if ``None``, cpu cores will be distributed evenly by all processes,\n        only take effect when ``num_processes`` > 1\n    :param channels_last: whether convert input to channels last memory formats,\n        defaults to ``False``.\n    :param auto_lr: whether to scale the learning rate linearly by ``num_processes`` times.\n        Defaults to ``True``.\n        If ``num_processes=1`` or other ``lr_scheduler`` is set, ``auto_lr`` will be ignored.\n    "
    if 'strategy' in kwargs:
        strategy = kwargs['strategy']
        if strategy == 'deepspeed' or isinstance(strategy, DeepSpeedStrategy):
            invalidInputError(False, 'bigdl.nano.pytorch.nano do not support deepspeed strategy')
    invalidInputError(distributed_backend != 'spawn', 'bigdl.nano.pytorch.nano do not support spawn')

    def decorator(func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(*inner_args, **inner_kwargs):
            if False:
                i = 10
                return i + 15
            return _DecoratedTorchNano(*args, num_processes=num_processes, use_ipex=use_ipex, distributed_backend=distributed_backend, precision=precision, cpu_for_each_process=cpu_for_each_process, channels_last=channels_last, auto_lr=auto_lr, **kwargs).train(func, *inner_args, **inner_kwargs)
        return wrapper
    return decorator