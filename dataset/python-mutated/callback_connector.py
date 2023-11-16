import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union
import lightning.pytorch as pl
from lightning.fabric.utilities.registry import _load_external_callbacks
from lightning.pytorch.callbacks import Callback, Checkpoint, ModelCheckpoint, ModelSummary, ProgressBar, RichProgressBar, TQDMProgressBar
from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.callbacks.rich_model_summary import RichModelSummary
from lightning.pytorch.callbacks.timer import Timer
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info
_log = logging.getLogger(__name__)

class _CallbackConnector:

    def __init__(self, trainer: 'pl.Trainer'):
        if False:
            for i in range(10):
                print('nop')
        self.trainer = trainer

    def on_trainer_init(self, callbacks: Optional[Union[List[Callback], Callback]], enable_checkpointing: bool, enable_progress_bar: bool, default_root_dir: Optional[str], enable_model_summary: bool, max_time: Optional[Union[str, timedelta, Dict[str, int]]]=None) -> None:
        if False:
            while True:
                i = 10
        self.trainer._default_root_dir = default_root_dir or os.getcwd()
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.trainer.callbacks = callbacks or []
        self._configure_checkpoint_callbacks(enable_checkpointing)
        self._configure_timer_callback(max_time)
        self._configure_progress_bar(enable_progress_bar)
        self._configure_model_summary_callback(enable_model_summary)
        self.trainer.callbacks.extend(_load_external_callbacks('lightning.pytorch.callbacks_factory'))
        _validate_callbacks_list(self.trainer.callbacks)
        self.trainer.callbacks = self._reorder_callbacks(self.trainer.callbacks)

    def _configure_checkpoint_callbacks(self, enable_checkpointing: bool) -> None:
        if False:
            i = 10
            return i + 15
        if self.trainer.checkpoint_callbacks:
            if not enable_checkpointing:
                raise MisconfigurationException('Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list.')
        elif enable_checkpointing:
            self.trainer.callbacks.append(ModelCheckpoint())

    def _configure_model_summary_callback(self, enable_model_summary: bool) -> None:
        if False:
            while True:
                i = 10
        if not enable_model_summary:
            return
        model_summary_cbs = [type(cb) for cb in self.trainer.callbacks if isinstance(cb, ModelSummary)]
        if model_summary_cbs:
            rank_zero_info(f'Trainer already configured with model summary callbacks: {model_summary_cbs}. Skipping setting a default `ModelSummary` callback.')
            return
        progress_bar_callback = self.trainer.progress_bar_callback
        is_progress_bar_rich = isinstance(progress_bar_callback, RichProgressBar)
        model_summary: ModelSummary
        if progress_bar_callback is not None and is_progress_bar_rich:
            model_summary = RichModelSummary()
        else:
            model_summary = ModelSummary()
        self.trainer.callbacks.append(model_summary)

    def _configure_progress_bar(self, enable_progress_bar: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        progress_bars = [c for c in self.trainer.callbacks if isinstance(c, ProgressBar)]
        if len(progress_bars) > 1:
            raise MisconfigurationException('You added multiple progress bar callbacks to the Trainer, but currently only one progress bar is supported.')
        if len(progress_bars) == 1:
            if enable_progress_bar:
                return
            progress_bar_callback = progress_bars[0]
            raise MisconfigurationException(f'Trainer was configured with `enable_progress_bar=False` but found `{progress_bar_callback.__class__.__name__}` in callbacks list.')
        if enable_progress_bar:
            progress_bar_callback = TQDMProgressBar()
            self.trainer.callbacks.append(progress_bar_callback)

    def _configure_timer_callback(self, max_time: Optional[Union[str, timedelta, Dict[str, int]]]=None) -> None:
        if False:
            return 10
        if max_time is None:
            return
        if any((isinstance(cb, Timer) for cb in self.trainer.callbacks)):
            rank_zero_info('Ignoring `Trainer(max_time=...)`, callbacks list already contains a Timer.')
            return
        timer = Timer(duration=max_time, interval='step')
        self.trainer.callbacks.append(timer)

    def _attach_model_logging_functions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        lightning_module = self.trainer.lightning_module
        for callback in self.trainer.callbacks:
            callback.log = lightning_module.log
            callback.log_dict = lightning_module.log_dict

    def _attach_model_callbacks(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Attaches the callbacks defined in the model.\n\n        If a callback returned by the model's configure_callback method has the same type as one or several\n        callbacks already present in the trainer callbacks list, it will replace them.\n        In addition, all :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks\n        will be pushed to the end of the list, ensuring they run last.\n\n        "
        trainer = self.trainer
        model_callbacks = call._call_lightning_module_hook(trainer, 'configure_callbacks')
        if not model_callbacks:
            return
        model_callbacks = [model_callbacks] if not isinstance(model_callbacks, Sequence) else model_callbacks
        model_callback_types = {type(c) for c in model_callbacks}
        trainer_callback_types = {type(c) for c in trainer.callbacks}
        trainer_callback_types.discard(Callback)
        override_types = set()
        for model_cb in model_callback_types:
            for trainer_cb in trainer_callback_types:
                if issubclass(model_cb, trainer_cb):
                    override_types.add(trainer_cb)
                    break
        if override_types:
            rank_zero_info(f"The following callbacks returned in `LightningModule.configure_callbacks` will override existing callbacks passed to Trainer: {', '.join(sorted((t.__name__ for t in override_types)))}")
        all_callbacks = [c for c in trainer.callbacks if type(c) not in override_types]
        all_callbacks.extend(model_callbacks)
        all_callbacks = _CallbackConnector._reorder_callbacks(all_callbacks)
        trainer.callbacks = all_callbacks

    @staticmethod
    def _reorder_callbacks(callbacks: List[Callback]) -> List[Callback]:
        if False:
            for i in range(10):
                print('nop')
        'Moves all the tuner specific callbacks at the beginning of the list and all the `ModelCheckpoint` callbacks\n        to the end of the list. The sequential order within the group of checkpoint callbacks is preserved, as well as\n        the order of all other callbacks.\n\n        Args:\n            callbacks: A list of callbacks.\n\n        Return:\n            A new list in which the first elements are tuner specific callbacks and last elements are ModelCheckpoints\n            if there were any present in the input.\n\n        '
        tuner_callbacks: List[Callback] = []
        other_callbacks: List[Callback] = []
        checkpoint_callbacks: List[Callback] = []
        for cb in callbacks:
            if isinstance(cb, (BatchSizeFinder, LearningRateFinder)):
                tuner_callbacks.append(cb)
            elif isinstance(cb, Checkpoint):
                checkpoint_callbacks.append(cb)
            else:
                other_callbacks.append(cb)
        return tuner_callbacks + other_callbacks + checkpoint_callbacks

def _validate_callbacks_list(callbacks: List[Callback]) -> None:
    if False:
        print('Hello World!')
    stateful_callbacks = [cb for cb in callbacks if is_overridden('state_dict', instance=cb)]
    seen_callbacks = set()
    for callback in stateful_callbacks:
        if callback.state_key in seen_callbacks:
            raise RuntimeError(f'Found more than one stateful callback of type `{type(callback).__name__}`. In the current configuration, this callback does not support being saved alongside other instances of the same type. Please consult the documentation of `{type(callback).__name__}` regarding valid settings for the callback state to be checkpointable. HINT: The `callback.state_key` must be unique among all callbacks in the Trainer.')
        seen_callbacks.add(callback.state_key)