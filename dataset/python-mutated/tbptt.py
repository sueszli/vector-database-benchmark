import collections.abc as collections
from typing import Callable, Mapping, Optional, Sequence, Union
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from ignite.engine import _prepare_batch, Engine, EventEnum
from ignite.utils import apply_to_tensor

class Tbptt_Events(EventEnum):
    """Aditional tbptt events.

    Additional events for truncated backpropagation throught time dedicated
    trainer.
    """
    TIME_ITERATION_STARTED = 'time_iteration_started'
    TIME_ITERATION_COMPLETED = 'time_iteration_completed'

def _detach_hidden(hidden: Union[torch.Tensor, Sequence, Mapping, str, bytes]) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    if False:
        print('Hello World!')
    'Cut backpropagation graph.\n\n    Auxillary function to cut the backpropagation graph by detaching the hidden\n    vector.\n    '
    return apply_to_tensor(hidden, torch.Tensor.detach)

def create_supervised_tbptt_trainer(model: nn.Module, optimizer: Optimizer, loss_fn: nn.Module, tbtt_step: int, dim: int=0, device: Optional[str]=None, non_blocking: bool=False, prepare_batch: Callable=_prepare_batch) -> Engine:
    if False:
        return 10
    "Create a trainer for truncated backprop through time supervised models.\n\n    Training recurrent model on long sequences is computationally intensive as\n    it requires to process the whole sequence before getting a gradient.\n    However, when the training loss is computed over many outputs\n    (`X to many <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_),\n    there is an opportunity to compute a gradient over a subsequence. This is\n    known as\n    `truncated backpropagation through time <https://machinelearningmastery.com/\n    gentle-introduction-backpropagation-time/>`_.\n    This supervised trainer apply gradient optimization step every `tbtt_step`\n    time steps of the sequence, while backpropagating through the same\n    `tbtt_step` time steps.\n\n    Args:\n        model: the model to train.\n        optimizer: the optimizer to use.\n        loss_fn: the loss function to use.\n        tbtt_step: the length of time chunks (last one may be smaller).\n        dim: axis representing the time dimension.\n        device: device type specification (default: None).\n            Applies to batches.\n        non_blocking: if True and this copy is between CPU and GPU,\n            the copy may occur asynchronously with respect to the host. For other cases,\n            this argument has no effect.\n        prepare_batch: function that receives `batch`, `device`,\n            `non_blocking` and outputs tuple of tensors `(batch_x, batch_y)`.\n\n    Returns:\n        a trainer engine with supervised update function.\n\n    .. warning::\n\n        The internal use of `device` has changed.\n        `device` will now *only* be used to move the input data to the correct device.\n        The `model` should be moved by the user before creating an optimizer.\n\n        For more information see:\n\n        * `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_\n        * `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_\n    "

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> float:
        if False:
            while True:
                i = 10
        loss_list = []
        hidden = None
        (x, y) = batch
        for batch_t in zip(x.split(tbtt_step, dim=dim), y.split(tbtt_step, dim=dim)):
            (x_t, y_t) = prepare_batch(batch_t, device=device, non_blocking=non_blocking)
            engine.fire_event(Tbptt_Events.TIME_ITERATION_STARTED)
            model.train()
            optimizer.zero_grad()
            if hidden is None:
                (y_pred_t, hidden) = model(x_t)
            else:
                hidden = _detach_hidden(hidden)
                (y_pred_t, hidden) = model(x_t, hidden)
            loss_t = loss_fn(y_pred_t, y_t)
            loss_t.backward()
            optimizer.step()
            engine.state.output = loss_t.item()
            loss_list.append(loss_t.item())
            engine.fire_event(Tbptt_Events.TIME_ITERATION_COMPLETED)
        return sum(loss_list) / len(loss_list)
    engine = Engine(_update)
    engine.register_events(*Tbptt_Events)
    return engine