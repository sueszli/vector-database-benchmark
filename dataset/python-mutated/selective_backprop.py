"""Core SelectiveBackprop class and functions."""
from typing import Any, Callable, Tuple, Union
import warnings
import numpy as np
from bigdl.nano.utils.common import invalidInputError
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

def should_selective_backprop(current_duration: float, batch_idx: int, start: float=0.5, end: float=0.9, interrupt: int=2) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Decides if selective backprop should be run based on time in training.\n\n    Returns true if the ``current_duration`` is between ``start`` and\n    ``end``. It is recommended that SB be applied during the later stages of\n    a training run, once the model has already "learned" easy examples.\n\n    To preserve convergence, SB can be interrupted with vanilla minibatch\n    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be\n    used at every step during the SB interval. When ``interrupt=2``, SB will\n    alternate with vanilla minibatch steps.\n\n    Args:\n        current_duration (float): The elapsed training duration. Must be\n            within ``[0.0, 1.0)``.\n        batch_idx (int): The current batch within the epoch.\n        start (float, optional): The duration at which selective backprop\n            should be enabled, as a percentage. Default: ``0.5``.\n        end (float, optional): The duration at which selective backprop\n            should be disabled. Default: ``0.9``.\n        interrupt (int, optional): The number of batches between vanilla\n            minibatch gradient updates. Default: ``2``.\n\n    Returns\n    -------\n        bool: If selective backprop should be performed on this batch.\n\n    '
    is_interval = current_duration >= start and current_duration < end
    is_step = interrupt == 0 or (batch_idx + 1) % interrupt != 0
    return is_interval and is_step

def select_using_loss(batch: Union[torch.Tensor, torch.Tensor], batch_idx: int, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', keep: float=0.5, scale_factor: float=1, loss_fn: Callable=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        print('Hello World!')
    "\n    Prunes minibatches as a subroutine of :class:`.SelectiveBackprop`.\n\n    Computes the loss function on the provided training examples and runs minibatches\n    according to the difficulty. The fraction of the minibatch that is kept for gradient\n    computation is specified by the argument ``0 <= keep <= 1``.\n\n    To speed up SB's selection forward pass, the argument ``scale_factor`` can\n    be used to spatially downsample input tensors. The full-sized inputs\n    will still be used for the weight gradient computation.\n\n    Args:\n        batch (Union[torch.Tensor, torch.Tensor]): Batch to prune.\n        batch_idx (int): Index of the batch.\n        trainer (pl.Trainer): Current trainer, used for getting current model.\n        pl_module (pl.LightningModule): Current module, used for getting loss function.\n        keep (float, optional): Fraction of examples in the batch to keep. Default: ``0.5``.\n        scale_factor (float, optional): Multiplier between 0 and 1 for spatial size. Downsampling\n            requires the input tensor to be at least 3D. Default: ``1``.\n        loss_fn (Callable): Loss function of the form\n            ``loss(outputs, targets, reduction='none')``.\n            The function must take the keyword argument ``reduction='none'``\n            to ensure that per-sample losses are returned.\n\n    Returns\n    -------\n        (torch.Tensor, torch.Tensor): The pruned batch of inputs and targets\n\n    "
    INTERPOLATE_MODES = {3: 'linear', 4: 'bilinear', 5: 'trilinear'}
    (input, target) = (batch[0], batch[1])
    interp_mode = 'bilinear'
    if scale_factor > 1:
        invalidInputError(False, 'scale_factor must be <= 1')
    if scale_factor != 1:
        if input.dim() not in INTERPOLATE_MODES:
            invalidInputError(False, f'Input must be 3D, 4D,                 or 5D if scale_factor != 1, got {input.dim()}')
        interp_mode = INTERPOLATE_MODES[input.dim()]
    with torch.no_grad():
        N = input.shape[0]
        if scale_factor < 1:
            X_scaled = F.interpolate(input, scale_factor=scale_factor, mode=interp_mode, align_corners=False, recompute_scale_factor=False)
        else:
            X_scaled = input
        if loss_fn is None:
            invalidInputError(False, 'loss_fn must be passed explicitly to the class.')
        else:
            losses = loss_fn(trainer.model(input), target)
        if not len(losses) == len(target):
            invalidInputError(False, 'Losses have wrong dimension,             maybe they are reduced.             Please offer unreduced losses which have the same dimension with batch_size.             It can be passed by ``loss_fn=`` when you initialize the class.')
        sorted_idx = torch.argsort(torch.Tensor(losses))
        n_select = int(keep * N)
        percs = np.arange(0.5, N, 1) / N
        probs = percs ** (1.0 / keep - 1.0)
        probs = probs / np.sum(probs)
        select_percs_idx = np.random.choice(N, n_select, replace=False, p=probs)
        select_idx = sorted_idx[list(select_percs_idx)]
    return (input[select_idx], target[select_idx])

class SelectiveBackprop(Callback):
    """
    Selectively backpropagate gradients from a subset of each batch.

    Based on (`Jiang et al, 2019`_), Selective Backprop (SB) prunes minibatches
    according to the difficulty of the individual training examples, and only
    computes weight gradients over the pruned subset, reducing iteration time, and
    speeding up training.

    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input image tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    .. _Jiang et al, 2019: https://arxiv.org/abs/1910.00762

    Args:
    ----
        start (float, optional): SB interval start as fraction of training duration.
            Default: ``0.5``.
        end (float, optional): SB interval end as fraction of training duration.
            Default: ``0.9``.
        keep (float, optional): fraction of minibatch to select and keep for gradient computation.
            Default: ``0.5``.
        scale_factor (float, optional): scale for downsampling input for selection forward pass.
            Default: ``1.``.
        interrupt (int, optional): interrupt SB with a vanilla minibatch step every
            ``interrupt`` batches. Default: ``2``.
        loss_fn (Callable): Loss function of the form
            ``loss(outputs, targets, reduction='none')``.
            The function must take the keyword argument ``reduction='none'``
            to ensure that per-sample losses are returned.

    Example:
    -------
        .. testcode::

            from bigdl.nano.pytorch.algorithms.selective_backprop import SelectiveBackprop
            from bigdl.nano.pytorch import Trainer
            from torch import nn
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            sb = SelectiveBackprop(start=0.5, end=0.9, keep=0.5, loss_fn=loss_fn)
            trainer = Trainer(
                algorithms=[sb],
            )

    """

    def __init__(self, start: float=0.5, end: float=0.9, keep: float=0.5, scale_factor: float=1.0, interrupt: int=2, loss_fn: Callable=None):
        if False:
            i = 10
            return i + 15
        "\n        Selectively backpropagate gradients from a subset of each batch.\n\n        :param start: SB interval start as fraction of training duration.\n            Default: ``0.5``.\n        :param end: SB interval end as fraction of training duration.\n            Default: ``0.9``.\n        :param keep: fraction of minibatch to select and keep for gradient computation.\n            Default: ``0.5``.\n        :param scale_factor: scale for downsampling input for selection forward pass.\n            Default: ``1.``.\n        :param interrupt: interrupt SB with a vanilla minibatch step every\n            ``interrupt`` batches. Default: ``2``.\n        :param loss_fn: Loss function of the form\n            ``loss(outputs, targets, reduction='none')``.\n            The function must take the keyword argument ``reduction='none'``\n            to ensure that per-sample losses are returned.\n        "
        self.start = start
        self.end = end
        self.keep = keep
        self.scale_factor = scale_factor
        self.interrupt = interrupt
        self._loss_fn = loss_fn

    def __match(self, trainer: 'pl.Trainer', batch_idx: int) -> bool:
        if False:
            return 10
        is_keep = self.keep < 1
        if not is_keep:
            return False
        if trainer.max_epochs is None:
            warnings.warn("Cannot get trainer.max_epochs information,                 selective_backprop's start and end control will not work.                 0.5 will be used as training progress forever.")
            elapsed_duration = 0.5
        else:
            elapsed_duration = float(trainer.current_epoch) / float(trainer.max_epochs)
        is_chosen = should_selective_backprop(current_duration=float(elapsed_duration), batch_idx=batch_idx, start=self.start, end=self.end, interrupt=self.interrupt)
        return is_chosen

    def on_train_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', batch: Union[torch.Tensor, torch.Tensor], batch_idx: int, unused: Any=0):
        if False:
            i = 10
            return i + 15
        'Add PyTorch Lightning callback.'
        if self.__match(trainer, batch_idx):
            (input, target) = (batch[0], batch[1])
            if not isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor):
                invalidInputError(False, 'Multiple tensors                     not supported for this method yet.')
            (input, target) = select_using_loss(batch, batch_idx, trainer, pl_module, self.keep, self.scale_factor, self._loss_fn)
            batch[0] = input
            batch[1] = target