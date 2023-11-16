from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from typing_extensions import override
import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies.strategy import Strategy, TBroadcast

class SingleDeviceStrategy(Strategy):
    """Strategy that handles communication on a single device."""
    strategy_name = 'single_device'

    def __init__(self, device: _DEVICE='cpu', accelerator: pl.accelerators.accelerator.Accelerator | None=None, checkpoint_io: CheckpointIO | None=None, precision_plugin: Precision | None=None):
        if False:
            print('Hello World!')
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self._root_device = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    @override
    def reduce(self, tensor: Any | Tensor, *args: Any, **kwargs: Any) -> Any | Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Reduces a tensor from several distributed processes to one aggregated tensor. Since this strategy only\n        operates with a single device, the reduction is simply the identity.\n\n        Args:\n            tensor: the tensor to sync and reduce\n            *args: ignored\n            **kwargs: ignored\n\n        Return:\n            the unmodified input as reduction is not needed for single process operation\n\n        '
        return tensor

    @override
    def all_gather(self, tensor: Tensor, group: Any | None=None, sync_grads: bool=False) -> Tensor:
        if False:
            while True:
                i = 10
        'Perform a all_gather on all processes.'
        return tensor

    @property
    @override
    def root_device(self) -> torch.device:
        if False:
            while True:
                i = 10
        return self._root_device

    @override
    def model_to_device(self) -> None:
        if False:
            print('Hello World!')
        assert self.model is not None, 'self.model must be set before self.model.to()'
        self.model.to(self.root_device)

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        if False:
            print('Hello World!')
        self.model_to_device()
        super().setup(trainer)

    @property
    @override
    def is_global_zero(self) -> bool:
        if False:
            return 10
        return True

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        pass

    @override
    def broadcast(self, obj: TBroadcast, src: int=0) -> TBroadcast:
        if False:
            for i in range(10):
                print('nop')
        return obj

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if False:
            return 10
        strategy_registry.register(cls.strategy_name, cls, description=cls.__name__)