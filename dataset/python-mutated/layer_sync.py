from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module

class LayerSync(ABC):
    """Abstract base class for creating plugins that wrap layers of a model with synchronization logic for
    multiprocessing."""

    @abstractmethod
    def apply(self, model: Module) -> Module:
        if False:
            return 10
        'Override this method to apply synchronization to the layers of this model.'

    @abstractmethod
    def revert(self, model: Module) -> Module:
        if False:
            return 10
        'Override this method to undo all modifications made in :meth:`apply`.'

class TorchSyncBatchNorm(LayerSync):
    """A plugin that wraps all batch normalization layers of a model with synchronization logic for multiprocessing.

    This plugin has no effect in single-device operation.

    """

    def apply(self, model: Module) -> Module:
        if False:
            return 10
        'Add global batchnorm for a model spread across multiple GPUs and nodes.\n\n        Override this method to synchronize batchnorm layers between specific process groups instead\n        of the whole world.\n\n        Args:\n            model: Reference to the current LightningModule\n\n        Return:\n            LightningModule with batchnorm layers synchronized within the process groups.\n\n        '
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def revert(self, model: Module) -> Module:
        if False:
            print('Hello World!')
        'Convert the wrapped batchnorm layers back to regular batchnorm layers.\n\n        Args:\n            model: Reference to the current LightningModule\n\n        Return:\n            LightningModule with regular batchnorm layers that will no longer sync across processes.\n\n        '
        converted_module = model
        if isinstance(model, torch.nn.modules.batchnorm.SyncBatchNorm):
            converted_module = _BatchNormXd(model.num_features, model.eps, model.momentum, model.affine, model.track_running_stats)
            if model.affine:
                with torch.no_grad():
                    converted_module.weight = model.weight
                    converted_module.bias = model.bias
            converted_module.running_mean = model.running_mean
            converted_module.running_var = model.running_var
            converted_module.num_batches_tracked = model.num_batches_tracked
            if hasattr(model, 'qconfig'):
                converted_module.qconfig = model.qconfig
        for (name, child) in model.named_children():
            converted_module.add_module(name, self.revert(child))
        del model
        return converted_module

class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input: Tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        return