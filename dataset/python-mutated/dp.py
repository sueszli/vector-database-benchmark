from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import TBroadcast, TReduce
from lightning.fabric.utilities.apply_func import apply_to_collection
from lightning.fabric.utilities.distributed import ReduceOp

class DataParallelStrategy(ParallelStrategy):
    """Implements data-parallel training in a single process, i.e., the model gets replicated to each device and each
    gets a split of the data."""

    def __init__(self, accelerator: Optional[Accelerator]=None, parallel_devices: Optional[List[torch.device]]=None, checkpoint_io: Optional[CheckpointIO]=None, precision: Optional[Precision]=None):
        if False:
            print('Hello World!')
        super().__init__(accelerator=accelerator, parallel_devices=parallel_devices, cluster_environment=None, checkpoint_io=checkpoint_io, precision=precision)

    @property
    def root_device(self) -> torch.device:
        if False:
            i = 10
            return i + 15
        assert self.parallel_devices is not None
        return self.parallel_devices[0]

    @property
    def distributed_sampler_kwargs(self) -> None:
        if False:
            while True:
                i = 10
        return None

    def setup_module(self, module: Module) -> DataParallel:
        if False:
            return 10
        'Wraps the given model into a :class:`~torch.nn.DataParallel` module.'
        return DataParallel(module=module, device_ids=self.parallel_devices)

    def module_to_device(self, module: Module) -> None:
        if False:
            i = 10
            return i + 15
        module.to(self.root_device)

    def batch_to_device(self, batch: Any, device: Optional[torch.device]=None) -> Any:
        if False:
            while True:
                i = 10
        return batch

    def all_reduce(self, collection: TReduce, group: Optional[Any]=None, reduce_op: Optional[Union[ReduceOp, str]]='mean') -> TReduce:
        if False:
            for i in range(10):
                print('nop')

        def mean(t: Tensor) -> Tensor:
            if False:
                print('Hello World!')
            original_dtype = t.dtype
            return t.float().mean().to(original_dtype)
        return apply_to_collection(collection, Tensor, mean)

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def broadcast(self, obj: TBroadcast, src: int=0) -> TBroadcast:
        if False:
            print('Hello World!')
        return obj

    def reduce_boolean_decision(self, decision: bool, all: bool=True) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return decision

    def get_module_state_dict(self, module: Module) -> Dict[str, Union[Any, Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(module, DataParallel):
            module = module.module
        return super().get_module_state_dict(module)

    def load_module_state_dict(self, module: Module, state_dict: Dict[str, Union[Any, Tensor]], strict: bool=True) -> None:
        if False:
            while True:
                i = 10
        if isinstance(module, DataParallel):
            module = module.module
        super().load_module_state_dict(module=module, state_dict=state_dict, strict=strict)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if False:
            while True:
                i = 10
        strategy_registry.register('dp', cls, description=cls.__name__)