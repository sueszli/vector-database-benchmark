from __future__ import annotations
import warnings
from typing import Any, Iterable, List, cast, TYPE_CHECKING
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer
import nni.nas.nn.pytorch as nas_nn
from nni.nas.evaluator.pytorch import LightningModule, Trainer
from nni.mutable import Sample
from .supermodule.base import BaseSuperNetModule
if TYPE_CHECKING:
    from pytorch_lightning.core.optimizer import LightningOptimizer
__all__ = ['BaseSuperNetModule', 'BaseOneShotLightningModule']

class BaseOneShotLightningModule(LightningModule):
    _inner_module_note = "inner_module : pytorch_lightning.LightningModule\n        It's a `LightningModule <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`__\n        that defines computations, train/val loops, optimizers in a single class.\n        When used in NNI, the ``inner_module`` is the combination of instances of evaluator + base model\n        (to be precise, a base model wrapped with LightningModule in evaluator).\n    "
    __doc__ = "\n    The base class for all one-shot NAS modules.\n\n    :class:`BaseOneShotLightningModule` is implemented as a subclass of :class:`~nni.nas.evaluator.pytorch.Lightning`,\n    to be make it deceptively look like a lightning module to the trainer.\n    It's actually a wrapper of the lightning module in evaluator.\n    The composition of different lightning modules is as follows::\n\n        BaseOneShotLightningModule       <- Current class (one-shot logics)\n            |_ evaluator.LightningModule <- Part of evaluator (basic training logics)\n                |_ user's model          <- Model space, transformed to a supernet by current class.\n\n    The base class implemented several essential utilities,\n    such as preprocessing user's model, redirecting lightning hooks for user's model,\n    configuring optimizers and exporting NAS result are implemented in this class.\n\n    Attributes\n    ----------\n    training_module\n        PyTorch lightning module, which defines the training recipe (the lightning module part in evaluator).\n\n    Parameters\n    ----------\n    " + _inner_module_note
    trainer: Trainer

    @property
    def automatic_optimization(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def __init__(self, training_module: LightningModule):
        if False:
            print('Hello World!')
        super().__init__()
        self.training_module = training_module

    def supernet_modules(self) -> Iterable[BaseSuperNetModule]:
        if False:
            i = 10
            return i + 15
        'Return all supernet modules in the model space.'
        for module in self.modules():
            if isinstance(module, BaseSuperNetModule):
                yield module

    @property
    def model(self) -> nas_nn.ModelSpace:
        if False:
            print('Hello World!')
        "Return the model space defined by the user.\n\n        The model space is not guaranteed to have been transformed into a one-shot supernet.\n        For instance, when ``__init__`` hasn't completed, the model space will still be the original one.\n        "
        model = self.training_module.model
        if not isinstance(model, nas_nn.ModelSpace):
            raise TypeError(f'The model is expected to be a valid PyTorch model space, but got {type(model)}')
        return model

    def set_model(self, model: nn.Module) -> None:
        if False:
            print('Hello World!')
        'Set the model space to be searched.'
        self.training_module.set_model(model)

    def resample(self) -> Sample:
        if False:
            while True:
                i = 10
        'Trigger the resample for each :meth:`supernet_modules`.\n        Sometimes (e.g., in differentiable cases), it does nothing.\n\n        Returns\n        -------\n        dict\n            Sampled architecture.\n        '
        result = {}
        for module in self.supernet_modules():
            result.update(module.resample(memo=result))
        return result

    def export(self) -> Sample:
        if False:
            return 10
        '\n        Export the NAS result, ideally the best choice of each :meth:`supernet_modules`.\n        You may implement an ``export`` method for your customized :meth:`supernet_modules`.\n\n        Returns\n        --------\n        dict\n            Keys are labels of mutables, and values are the choice indices of them.\n        '
        result = {}
        for module in self.supernet_modules():
            result.update(module.export(memo=result))
        return result

    def export_probs(self) -> Sample:
        if False:
            i = 10
            return i + 15
        '\n        Export the probability of every choice in the search space got chosen.\n\n        .. note:: If such method of some modules is not implemented, they will be simply ignored.\n\n        Returns\n        -------\n        dict\n            In most cases, keys are labels of the mutables, while values are a dict,\n            whose key is the choice and value is the probability of it being chosen.\n        '
        result = {}
        for module in self.supernet_modules():
            try:
                result.update(module.export_probs(memo=result))
            except NotImplementedError:
                warnings.warn('Some super-modules you have used did not implement export_probs. You might find some logs are missing.', UserWarning)
        return result

    def log_probs(self, probs: Sample) -> None:
        if False:
            return 10
        '\n        Write the probability of every choice to the logger.\n        (nothing related to log-probability stuff).\n\n        Parameters\n        ----------\n        probs\n            The result of :meth:`export_probs`.\n        '
        self.log_dict({f'prob/{label}/{value}': logit for (label, dist) in probs.items() for (value, logit) in dist.items()})

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.training_module(x)

    def configure_optimizers(self) -> Any:
        if False:
            return 10
        '\n        Transparently configure optimizers for the inner model,\n        unless one-shot algorithm has its own optimizer (via :meth:`configure_architecture_optimizers`),\n        in which case, the optimizer will be appended to the list.\n\n        The return value is still one of the 6 types defined in PyTorch-Lightning.\n        '
        arch_optimizers = self.configure_architecture_optimizers() or []
        if not arch_optimizers:
            return self.training_module.configure_optimizers()
        if isinstance(arch_optimizers, optim.Optimizer):
            arch_optimizers = [arch_optimizers]
        for optimizer in arch_optimizers:
            optimizer.is_arch_optimizer = True
        optim_conf: Any = self.training_module.configure_optimizers()
        optim_conf = self.postprocess_weight_optimizers(optim_conf)
        if optim_conf is None:
            return arch_optimizers
        if isinstance(optim_conf, Optimizer):
            return [optim_conf] + arch_optimizers
        if isinstance(optim_conf, (list, tuple)) and len(optim_conf) == 2 and isinstance(optim_conf[0], list) and all((isinstance(opt, Optimizer) for opt in optim_conf[0])):
            return (list(optim_conf[0]) + arch_optimizers, optim_conf[1])
        if isinstance(optim_conf, dict):
            return [optim_conf] + [{'optimizer': optimizer} for optimizer in arch_optimizers]
        if isinstance(optim_conf, (list, tuple)) and all((isinstance(d, dict) for d in optim_conf)):
            return list(optim_conf) + [{'optimizer': optimizer} for optimizer in arch_optimizers]
        if isinstance(optim_conf, (list, tuple)) and all((isinstance(opt, Optimizer) for opt in optim_conf)):
            return list(optim_conf) + arch_optimizers
        warnings.warn('Unknown optimizer configuration. Architecture optimizers will be ignored. Strategy might fail.', UserWarning)
        return optim_conf

    def setup(self, stage: str=cast(str, None)):
        if False:
            print('Hello World!')
        self.training_module.trainer = self.trainer
        self.training_module.log = self.log
        self._optimizer_progress = 0
        return self.training_module.setup(stage)

    def teardown(self, stage: str=cast(str, None)):
        if False:
            while True:
                i = 10
        return self.training_module.teardown(stage)

    def postprocess_weight_optimizers(self, optimizers: Any) -> Any:
        if False:
            return 10
        '\n        Some subclasss need to modify the original optimizers. This is where it should be done.\n        For example, differentiable algorithms might not want the architecture weights to be inside the weight optimizers.\n\n        Returns\n        -------\n        By default, it return the original object.\n        '
        return optimizers

    def configure_architecture_optimizers(self) -> list[optim.Optimizer] | optim.Optimizer | None:
        if False:
            print('Hello World!')
        '\n        Hook kept for subclasses. A specific NAS method inheriting this base class should return its architecture optimizers here\n        if architecture parameters are needed. Note that lr schedulers are not supported now for architecture_optimizers.\n\n        Returns\n        -------\n        Optimizers used by a specific NAS algorithm. Return None if no architecture optimizers are needed.\n        '
        return None

    def advance_optimization(self, loss: Any, batch_idx: int, gradient_clip_val: int | float | None=None, gradient_clip_algorithm: str | None=None):
        if False:
            print('Hello World!')
        '\n        Run the optimizer defined in evaluators, when manual optimization is turned on.\n\n        Call this method when the model should be optimized.\n        To keep it as neat as possible, we only implement the basic ``zero_grad``, ``backward``, ``grad_clip``, and ``step`` here.\n        Many hooks and pre/post-processing are omitted.\n        Inherit this method if you need more advanced behavior.\n\n        The full optimizer step could be found\n        `here <https://github.com/Lightning-AI/lightning/blob/0e531283/src/pytorch_lightning/loops/optimization/optimizer_loop.py>`__.\n        We only implement part of the optimizer loop here.\n\n        Parameters\n        ----------\n        batch_idx: int\n            The current batch index.\n        '
        if self.automatic_optimization:
            raise ValueError('This method should not be used when automatic optimization is turned on.')
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        optimizers = cast(List[Optimizer], [opt for opt in optimizers if not getattr(opt, 'is_arch_optimizer', False)])
        if hasattr(self.trainer, 'optimizer_frequencies'):
            self._legacy_advance_optimization(loss, batch_idx, optimizers, gradient_clip_val, gradient_clip_algorithm)
        else:
            if not self.training_module.automatic_optimization:
                raise ValueError('Evaluator module with manual optimization is not compatible with one-shot algorithms.')
            if len(optimizers) != 1:
                raise ValueError('More than one optimizer returned by evaluator. This is not supported in NAS.')
            optimizer = optimizers[0]
            self.training_module.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer)
            self.manual_backward(loss)
            self.training_module.configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
            self.training_module.optimizer_step(self.trainer.current_epoch, batch_idx, optimizer)
        self._optimizer_progress += 1

    def _legacy_advance_optimization(self, loss: Any, batch_idx: int, optimizers: list[Optimizer], gradient_clip_val: int | float | None=None, gradient_clip_algorithm: str | None=None):
        if False:
            print('Hello World!')
        ':meth:`advance_optimization` for Lightning 1.x.'
        if self.trainer.optimizer_frequencies:
            warnings.warn('optimizer_frequencies is not supported in NAS. It will be ignored.', UserWarning)
        opt_idx = self._optimizer_progress % len(optimizers)
        optimizer = cast(Optimizer, optimizers[opt_idx])
        self.training_module.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)
        self.manual_backward(loss)
        self.training_module.configure_gradient_clipping(optimizer, opt_idx, gradient_clip_val, gradient_clip_algorithm)
        self.training_module.optimizer_step(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)

    def advance_lr_schedulers(self, batch_idx: int):
        if False:
            return 10
        '\n        Advance the learning rates, when manual optimization is turned on.\n\n        The full implementation is\n        `here <https://github.com/Lightning-AI/lightning/blob/0e531283/src/pytorch_lightning/loops/epoch/training_epoch_loop.py>`__.\n        We only include a partial implementation here.\n        Advanced features like Reduce-lr-on-plateau are not supported.\n        '
        if self.automatic_optimization:
            raise ValueError('This method should not be used when automatic optimization is turned on.')
        self._advance_lr_schedulers_impl(batch_idx, 'step')
        if self.trainer.is_last_batch:
            self._advance_lr_schedulers_impl(batch_idx, 'epoch')

    def _advance_lr_schedulers_impl(self, batch_idx: int, interval: str):
        if False:
            for i in range(10):
                print('nop')
        current_idx = batch_idx if interval == 'step' else self.trainer.current_epoch
        current_idx += 1
        try:
            for config in self.trainer.lr_scheduler_configs:
                if hasattr(config, 'opt_idx'):
                    (scheduler, opt_idx) = (config.scheduler, config.opt_idx)
                else:
                    (scheduler, opt_idx) = (config.scheduler, None)
                if config.reduce_on_plateau:
                    warnings.warn('Reduce-lr-on-plateau is not supported in NAS. It will be ignored.', UserWarning)
                if config.interval == interval and current_idx % config.frequency == 0:
                    if opt_idx is not None:
                        self.training_module.lr_scheduler_step(cast(Any, scheduler), cast(int, opt_idx), None)
                    else:
                        self.training_module.lr_scheduler_step(cast(Any, scheduler), None)
        except AttributeError:
            for lr_scheduler in self.trainer.lr_schedulers:
                if lr_scheduler['reduce_on_plateau']:
                    warnings.warn('Reduce-lr-on-plateau is not supported in NAS. It will be ignored.', UserWarning)
                if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency']:
                    lr_scheduler['scheduler'].step()

    def architecture_optimizers(self) -> list[LightningOptimizer] | LightningOptimizer | None:
        if False:
            print('Hello World!')
        '\n        Get the optimizers configured in :meth:`configure_architecture_optimizers`.\n\n        Return type would be LightningOptimizer or list of LightningOptimizer.\n        '
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        optimizers = [opt for opt in optimizers if getattr(opt, 'is_arch_optimizer', False)]
        if not optimizers:
            return None
        if len(optimizers) == 1:
            return optimizers[0]
        return optimizers

    def on_train_start(self):
        if False:
            return 10
        return self.training_module.on_train_start()

    def on_train_end(self):
        if False:
            for i in range(10):
                print('nop')
        return self.training_module.on_train_end()

    def on_validation_start(self):
        if False:
            return 10
        return self.training_module.on_validation_start()

    def on_validation_end(self):
        if False:
            return 10
        return self.training_module.on_validation_end()

    def on_fit_start(self):
        if False:
            return 10
        return self.training_module.on_fit_start()

    def on_fit_end(self):
        if False:
            while True:
                i = 10
        return self.training_module.on_fit_end()

    def on_train_batch_start(self, batch, batch_idx, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.training_module.on_train_batch_start(batch, batch_idx, *args, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx, *args, **kwargs):
        if False:
            return 10
        return self.training_module.on_train_batch_end(outputs, batch, batch_idx, *args, **kwargs)

    def on_train_epoch_start(self):
        if False:
            for i in range(10):
                print('nop')
        return self.training_module.on_train_epoch_start()

    def on_train_epoch_end(self):
        if False:
            print('Hello World!')
        return self.training_module.on_train_epoch_end()

    def on_before_backward(self, loss):
        if False:
            i = 10
            return i + 15
        return self.training_module.on_before_backward(loss)

    def on_after_backward(self):
        if False:
            print('Hello World!')
        return self.training_module.on_after_backward()