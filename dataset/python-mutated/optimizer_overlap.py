from abc import ABC, abstractmethod
import inspect
from typing import Dict, Type
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.distributed.optim import as_functional_optim
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import _OptimizerHookState, _hook_then_optimizer
_registered_overlapped_optims: Dict[Type, Type] = {}

def register_overlapped(optim_cls):
    if False:
        while True:
            i = 10

    def decorator(target_overlapped_optim_cls):
        if False:
            while True:
                i = 10
        if target_overlapped_optim_cls in _registered_overlapped_optims:
            raise ValueError(f'{target_overlapped_optim_cls} already registered with optim_cls {_registered_overlapped_optims[optim_cls]} {optim_cls}, trying tore-register it for {optim_cls} is not supported.')
        _registered_overlapped_optims[optim_cls] = target_overlapped_optim_cls
        return target_overlapped_optim_cls
    return decorator

class OverlappedOptimizer(ABC):

    def __init__(self, optim_cls: Type) -> None:
        if False:
            while True:
                i = 10
        '\n        Initialize the OverlappedOptimizer.\n\n        Overlappedoptimizer is a base class that child classes can implement to\n        specify how different optimizers will register themselves with DDP.\n        '
        self.optim_cls = optim_cls

    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        if False:
            while True:
                i = 10
        'Registers the overlapped optimizer with DDP.'
        raise NotImplementedError(f'{self.__class__.__name__} does not support overlapped DDP.')

    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Registers the overlapped optimizer with FSDP.'
        raise NotImplementedError(f'{self.__class__.__name__} does not support overlapped FSDP.')

@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: Type, params, *optim_args, **optim_kwargs) -> None:
        if False:
            return 10
        super().__init__(optim_cls)
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)
        self._opt_hook_state = _OptimizerHookState(f_optim, params)

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        if False:
            while True:
                i = 10
        ddp_inst.register_comm_hook(None, _hook_then_optimizer(allreduce_hook, self._opt_hook_state))

    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Register the overlapped optimizer with FSDP.'
        raise NotImplementedError(f'{self.__class__.__name__} does not support overlapped FSDP.')

def _as_overlapped_optim(optim_cls: Type, params, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``.'
    for clz in inspect.getmro(optim_cls):
        try:
            return _registered_overlapped_optims[clz](optim_cls, params, *args, **kwargs)
        except KeyError:
            pass
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)