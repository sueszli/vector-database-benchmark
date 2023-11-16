import operator
from contextlib import contextmanager
from functools import partial
from logging import warning
from typing import Any, Union, Callable
import torch
from torch.nn import Module
from torch.optim import Optimizer, LBFGS
import pytorch_lightning as pl
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import PrecisionPlugin
from bigdl.nano.utils.common import compare_version
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.deps.ipex.ipex_api import ipex_optimize

class IPEXStrategy(SingleDeviceStrategy):
    """IPEX strategy."""
    strategy_name = 'ipex'

    def __init__(self, accelerator: Accelerator=None, precision_plugin: PrecisionPlugin=PrecisionPlugin(), dtype=None) -> None:
        if False:
            print('Hello World!')
        '\n        Create a IPEXStrategy.\n\n        :param accelerator: the accelerator to handle hardware\n        :param precision_plugin: the plugin to handle precision-specific parts\n        '
        self.dtype = dtype
        if self.dtype == torch.bfloat16 and isinstance(precision_plugin, PrecisionPlugin):
            precision_plugin = IPEXBF16Precision()
        super().__init__(accelerator=accelerator, precision_plugin=precision_plugin)

    def setup(self, trainer: pl.Trainer) -> None:
        if False:
            print('Hello World!')
        "\n        Setup plugins for the trainer fit and creates optimizers.\n\n        We add IPEX's optimizations here.\n\n        :param trainer: the trainer instance\n        "
        super().setup(trainer)
        ipex_optimize(self.model, optimizers=self.optimizers, inplace=True, dtype=self.dtype)

class IPEXBF16Precision(PrecisionPlugin):
    """Create Precision Plugin for IPEX BFloat16."""
    precision: Union[str, int] = 'bf16'

    @contextmanager
    def forward_context(self):
        if False:
            return 10
        'AMP for managing model forward/training_step/evaluation_step/predict_step.'
        with torch.cpu.amp.autocast():
            yield

    def optimizer_step(self, model: Union['pl.LightningModule', Module], optimizer: Optimizer, optimizer_idx: int, closure: Callable[[], Any], **kwargs: Any) -> Any:
        if False:
            return 10
        'Bf16 optimizer step.'
        if compare_version('intel_extension_for_pytorch', operator.lt, '1.13.100'):
            from intel_extension_for_pytorch.optim._optimizer_utils import IPEX_FUSED_OPTIMIZER_LIST
        else:
            from intel_extension_for_pytorch.optim._optimizer_utils import IPEX_FUSED_OPTIMIZER_LIST_CPU as IPEX_FUSED_OPTIMIZER_LIST
        'Hook to run the optimizer step.'
        if type(optimizer) in IPEX_FUSED_OPTIMIZER_LIST:
            return super().optimizer_step(model, optimizer, optimizer_idx, closure, **kwargs)
        if isinstance(model, pl.LightningModule):
            closure = partial(self._wrap_closure, model, optimizer, optimizer_idx, closure)
        if isinstance(optimizer, LBFGS):
            invalidInputError(False, f'IPEX BFloat16 and the LBFGS optimizer are not compatible (optimizer {optimizer_idx}', "Hint: Set 'use_ipex' to False or not set 'precision' to 'bf16' if LBFGS optimizer is necessary")
        if type(optimizer).__name__ not in dir(torch.optim):
            warning("Seems like you are using a custom optimizer,please make sure that 'optimizer.step(closure)' does not need to be called in training stage")
        closure_result = closure()
        optimizer.step(**kwargs)
        return closure_result