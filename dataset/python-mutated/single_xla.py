import os
from typing import Optional, Union
import torch
from typing_extensions import override
import lightning.pytorch as pl
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins import XLACheckpointIO
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision.xla import XLAPrecision
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters

class SingleDeviceXLAStrategy(SingleDeviceStrategy):
    """Strategy for training on a single XLA device."""

    def __init__(self, device: _DEVICE, accelerator: Optional['pl.accelerators.Accelerator']=None, checkpoint_io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]]=None, precision_plugin: Optional[XLAPrecision]=None, debug: bool=False):
        if False:
            return 10
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        if isinstance(device, torch.device):
            device = device.index
        import torch_xla.core.xla_model as xm
        super().__init__(accelerator=accelerator, device=xm.xla_device(device), checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self.debug = debug

    @property
    @override
    def checkpoint_io(self) -> Union[XLACheckpointIO, _WrappingCheckpointIO]:
        if False:
            return 10
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, (XLACheckpointIO, _WrappingCheckpointIO))
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]]) -> None:
        if False:
            return 10
        if io is not None and (not isinstance(io, (XLACheckpointIO, _WrappingCheckpointIO))):
            raise TypeError(f'The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}')
        self._checkpoint_io = io

    @property
    @override
    def precision_plugin(self) -> XLAPrecision:
        if False:
            for i in range(10):
                print('nop')
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, XLAPrecision)
            return plugin
        return XLAPrecision()

    @precision_plugin.setter
    @override
    def precision_plugin(self, precision_plugin: Optional[XLAPrecision]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if precision_plugin is not None and (not isinstance(precision_plugin, XLAPrecision)):
            raise TypeError(f'The XLA strategy can only work with the `XLAPrecision` plugin, found {precision_plugin}')
        self._precision_plugin = precision_plugin

    @override
    def setup(self, trainer: 'pl.Trainer') -> None:
        if False:
            return 10
        assert self.model, 'self.model must be set before find_shared_parameters(self.model)'
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        set_shared_parameters(self.model, shared_params)
        super().setup(trainer)
        if self.debug:
            os.environ['PT_XLA_DEBUG'] = str(1)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if False:
            while True:
                i = 10
        strategy_registry.register('single_xla', cls, description=cls.__name__)

    @override
    def teardown(self) -> None:
        if False:
            i = 10
            return i + 15
        super().teardown()
        os.environ.pop('PT_XLA_DEBUG', None)