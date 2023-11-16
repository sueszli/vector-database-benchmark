from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, OrderedDict
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override
import lightning.pytorch as pl
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.optimization.closure import AbstractClosure, OutputResult
from lightning.pytorch.loops.progress import _OptimizationProgress
from lightning.pytorch.loops.utilities import _block_parallel_sync_behavior
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache
from lightning.pytorch.utilities.types import STEP_OUTPUT

@dataclass
class ClosureResult(OutputResult):
    """A container to hold the result of a :class:`Closure` call.

    It is created from the output of :meth:`~lightning.pytorch.core.LightningModule.training_step`.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        extra: Any keys other than the loss returned.

    """
    closure_loss: Optional[Tensor]
    loss: Optional[Tensor] = field(init=False, default=None)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        self._clone_loss()

    def _clone_loss(self) -> None:
        if False:
            print('Hello World!')
        if self.closure_loss is not None:
            self.loss = self.closure_loss.detach().clone()

    @classmethod
    def from_training_step_output(cls, training_step_output: STEP_OUTPUT, normalize: int=1) -> 'ClosureResult':
        if False:
            return 10
        (closure_loss, extra) = (None, {})
        if isinstance(training_step_output, Mapping):
            closure_loss = training_step_output.get('loss')
            if closure_loss is None:
                raise MisconfigurationException("In automatic_optimization, when `training_step` returns a dict, the 'loss' key needs to be present")
            extra = {k: v for (k, v) in training_step_output.items() if k != 'loss'}
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output
        elif training_step_output is not None:
            raise MisconfigurationException('In automatic optimization, `training_step` must return a Tensor, a dict, or None (where the step will be skipped).')
        if closure_loss is not None:
            closure_loss = closure_loss / normalize
        return cls(closure_loss, extra=extra)

    @override
    def asdict(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'loss': self.loss, **self.extra}

class Closure(AbstractClosure[ClosureResult]):
    """An implementation of a :class:`AbstractClosure` for automatic optimization in Lightning that combines three
    elementary closures into one: ``training_step``, ``backward`` and ``zero_grad``.

    The Closure gets created by the training loop(s) and is then passed to the
    :meth:`torch.optim.Optimizer.step` method. An optimizer is responsible for calling the closure and optionally
    do something with the output.

    Args:
        step_fn: This is typically the :meth:`lightning.pytorch.core.module.LightningModule.training_step
            wrapped with processing for its outputs
        backward_fn: A function that takes a loss value as input, performs back-propagation and returns the loss value.
            Can be set to ``None`` to skip the backward operation.
        zero_grad_fn: A function that zeroes the gradients. Can be set to ``None`` to skip zero_grad, for example
            when accumulating gradients.

    Example:

        closure = Closure()
        optimizer = torch.optim.Adam(...)
        optimizer.step(closure)
    """
    warning_cache = WarningCache()

    def __init__(self, step_fn: Callable[[], ClosureResult], backward_fn: Optional[Callable[[Tensor], None]]=None, zero_grad_fn: Optional[Callable[[], None]]=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn

    @override
    @torch.enable_grad()
    def closure(self, *args: Any, **kwargs: Any) -> ClosureResult:
        if False:
            for i in range(10):
                print('nop')
        step_output = self._step_fn()
        if step_output.closure_loss is None:
            self.warning_cache.warn('`training_step` returned `None`. If this was on purpose, ignore this warning...')
        if self._zero_grad_fn is not None:
            self._zero_grad_fn()
        if self._backward_fn is not None and step_output.closure_loss is not None:
            self._backward_fn(step_output.closure_loss)
        return step_output

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
        if False:
            print('Hello World!')
        self._result = self.closure(*args, **kwargs)
        return self._result.loss
_OUTPUTS_TYPE = Dict[str, Any]

class _AutomaticOptimization(_Loop):
    """Performs automatic optimization (forward, zero grad, backward, optimizer step)"""
    output_result_cls = ClosureResult

    def __init__(self, trainer: 'pl.Trainer') -> None:
        if False:
            while True:
                i = 10
        super().__init__(trainer)
        self.optim_progress: _OptimizationProgress = _OptimizationProgress()
        self._skip_backward: bool = False

    def run(self, optimizer: Optimizer, batch_idx: int, kwargs: OrderedDict) -> _OUTPUTS_TYPE:
        if False:
            while True:
                i = 10
        'Runs closure (train step + backward) together with optimization if necessary.\n\n        Args:\n            kwargs: the kwargs passed down to the hooks\n            batch_idx: the current batch index.\n            optimizer: the optimizer\n\n        '
        closure = self._make_closure(kwargs, optimizer, batch_idx)
        if not self.trainer.strategy.handles_gradient_accumulation and self.trainer.fit_loop._should_accumulate():
            with _block_parallel_sync_behavior(self.trainer.strategy, block=True):
                closure()
        else:
            self._optimizer_step(batch_idx, closure)
        result = closure.consume_result()
        if result.loss is None:
            return {}
        return result.asdict()

    def _make_closure(self, kwargs: OrderedDict, optimizer: Optimizer, batch_idx: int) -> Closure:
        if False:
            i = 10
            return i + 15
        'Build a closure object that captures the given arguments and runs the `training_step` function and\n        optionally other functions such as `backward` and `zero_grad`.'
        step_fn = self._make_step_fn(kwargs)
        backward_fn = self._make_backward_fn(optimizer)
        zero_grad_fn = self._make_zero_grad_fn(batch_idx, optimizer)
        return Closure(step_fn=step_fn, backward_fn=backward_fn, zero_grad_fn=zero_grad_fn)

    def _make_step_fn(self, kwargs: OrderedDict) -> Callable[[], ClosureResult]:
        if False:
            print('Hello World!')
        'Build the step function that runs the `training_step` and processes its output.'
        return partial(self._training_step, kwargs)

    def _make_zero_grad_fn(self, batch_idx: int, optimizer: Optimizer) -> Optional[Callable[[], None]]:
        if False:
            return 10
        'Build a `zero_grad` function that zeroes the gradients before back-propagation.\n\n        Returns ``None`` in the case backward needs to be skipped.\n\n        '
        if self._skip_backward:
            return None
        is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
        if not is_first_batch_to_accumulate:
            return None

        def zero_grad_fn() -> None:
            if False:
                return 10
            self._on_before_zero_grad(optimizer)
            self._optimizer_zero_grad(batch_idx, optimizer)
        return zero_grad_fn

    def _make_backward_fn(self, optimizer: Optimizer) -> Optional[Callable[[Tensor], None]]:
        if False:
            i = 10
            return i + 15
        'Build a `backward` function that handles back-propagation through the output produced by the `training_step`\n        function.\n\n        Returns ``None`` in the case backward needs to be skipped.\n\n        '
        if self._skip_backward:
            return None

        def backward_fn(loss: Tensor) -> None:
            if False:
                i = 10
                return i + 15
            call._call_strategy_hook(self.trainer, 'backward', loss, optimizer)
        return backward_fn

    def _optimizer_step(self, batch_idx: int, train_step_and_backward_closure: Callable[[], Optional[Tensor]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Performs the optimizer step and some sanity checking.\n\n        Args:\n            batch_idx: the index of the current batch\n            train_step_and_backward_closure: the closure function performing the train step and computing the\n                gradients. By default, called by the optimizer (if possible)\n\n        '
        trainer = self.trainer
        optimizer = trainer.strategy._lightning_optimizers[0]
        should_accumulate = trainer.fit_loop._should_accumulate()
        if not should_accumulate:
            self.optim_progress.optimizer.step.increment_ready()
        call._call_lightning_module_hook(trainer, 'optimizer_step', trainer.current_epoch, batch_idx, optimizer, train_step_and_backward_closure)
        if not should_accumulate:
            self.optim_progress.optimizer.step.increment_completed()

    def _on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        if False:
            i = 10
            return i + 15
        'Calls the ``on_before_zero_grad`` hook.\n\n        Args:\n            optimizer: the current optimizer\n\n        '
        trainer = self.trainer
        self.optim_progress.optimizer.zero_grad.increment_ready()
        call._call_callback_hooks(trainer, 'on_before_zero_grad', optimizer)
        call._call_lightning_module_hook(trainer, 'on_before_zero_grad', optimizer)
        self.optim_progress.optimizer.zero_grad.increment_started()

    def _optimizer_zero_grad(self, batch_idx: int, optimizer: torch.optim.Optimizer) -> None:
        if False:
            i = 10
            return i + 15
        'Zeroes out all gradients of parameters optimized by the current optimizer.\n\n        Args:\n            batch_idx: the index of the current batch\n            optimizer: the current optimizer\n\n        '
        trainer = self.trainer
        call._call_lightning_module_hook(trainer, 'optimizer_zero_grad', trainer.current_epoch, batch_idx, optimizer)
        self.optim_progress.optimizer.zero_grad.increment_completed()

    def _training_step(self, kwargs: OrderedDict) -> ClosureResult:
        if False:
            print('Hello World!')
        'Performs the actual train step with the tied hooks.\n\n        Args:\n            kwargs: the kwargs passed down to the hooks.\n\n        Returns:\n            A ``ClosureResult`` containing the training step output.\n\n        '
        trainer = self.trainer
        training_step_output = call._call_strategy_hook(trainer, 'training_step', *kwargs.values())
        self.trainer.strategy.post_training_step()
        return self.output_result_cls.from_training_step_output(training_step_output, trainer.accumulate_grad_batches)