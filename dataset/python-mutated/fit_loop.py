import logging
from typing import Optional, Union
import torch
from typing_extensions import override
import lightning.pytorch as pl
from lightning.fabric.utilities.data import _set_sampler_epoch, sized_len
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.loops import _Loop
from lightning.pytorch.loops.fetchers import _DataFetcher
from lightning.pytorch.loops.progress import _Progress
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.utilities import _is_max_limit_reached, _select_data_fetcher
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.data_connector import _check_dataloader_iterable, _DataLoaderSource, _parse_num_batches, _process_dataloader, _request_dataloader, _resolve_overfit_batches
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.combined_loader import _SUPPORTED_MODES, CombinedLoader
from lightning.pytorch.utilities.data import has_len_all_ranks
from lightning.pytorch.utilities.exceptions import MisconfigurationException, SIGTERMException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_warn
log = logging.getLogger(__name__)

class _FitLoop(_Loop):
    """This loop is the top-level loop where training starts.

    It simply counts the epochs and iterates from one to the next by calling ``TrainingEpochLoop.run()`` in its
    ``advance()`` method.

    Example::

        # FitLoop
        for epoch in range(max_epochs):
            # TrainingEpochLoop
            for batch_idx, batch in enumerate(train_dataloader):
                loss = lightning_module.training_step(batch, batch_idx)
                ...

                # ValidationEpochLoop
                for batch_idx, batch in enumerate(val_dataloader):
                    lightning_module.validation_step(batch, batch_idx)
                    ...
                ...
            ...

    Args:
        min_epochs: The minimum number of epochs
        max_epochs: The maximum number of epochs, can be set -1 to turn this limit off

    """

    def __init__(self, trainer: 'pl.Trainer', min_epochs: Optional[int]=0, max_epochs: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(trainer)
        if isinstance(max_epochs, int) and max_epochs < -1:
            raise MisconfigurationException(f'`max_epochs` must be a non-negative integer or -1. You passed in {max_epochs}.')
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.epoch_loop = _TrainingEpochLoop(trainer)
        self.epoch_progress = _Progress()
        self.max_batches: Union[int, float] = float('inf')
        self._data_source = _DataLoaderSource(None, 'train_dataloader')
        self._combined_loader: Optional[CombinedLoader] = None
        self._data_fetcher: Optional[_DataFetcher] = None
        self._last_train_dl_reload_epoch = float('-inf')

    @property
    def total_batch_idx(self) -> int:
        if False:
            return 10
        'Returns the current batch index (across epochs)'
        return self.epoch_loop.total_batch_idx

    @property
    def batch_idx(self) -> int:
        if False:
            return 10
        'Returns the current batch index (within this epoch)'
        return self.epoch_loop.batch_idx

    @property
    def min_steps(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Returns the minimum number of steps to run.'
        return self.epoch_loop.min_steps

    @property
    def max_steps(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the maximum number of steps to run.'
        return self.epoch_loop.max_steps

    @_Loop.restarting.setter
    @override
    def restarting(self, restarting: bool) -> None:
        if False:
            i = 10
            return i + 15
        values = (self.epoch_progress.current.ready, self.epoch_progress.current.started)
        epoch_unfinished = any((v != self.epoch_progress.current.processed for v in values))
        restarting = restarting and epoch_unfinished or self._iteration_based_training()
        _Loop.restarting.fset(self, restarting)

    @property
    def _skip_backward(self) -> bool:
        if False:
            return 10
        'Determines whether the loop will skip backward during automatic optimization.'
        return self.epoch_loop.automatic_optimization._skip_backward

    @_skip_backward.setter
    def _skip_backward(self, value: bool) -> None:
        if False:
            print('Hello World!')
        'Determines whether the loop will skip backward during automatic optimization.'
        self.epoch_loop.automatic_optimization._skip_backward = value

    @property
    def _results(self) -> _ResultCollection:
        if False:
            i = 10
            return i + 15
        if self.trainer.training:
            return self.epoch_loop._results
        if self.trainer.validating:
            return self.epoch_loop.val_loop._results
        raise RuntimeError("`FitLoop._results` property isn't defined. Accessed outside of scope")

    @property
    def _can_stop_early(self) -> bool:
        if False:
            i = 10
            return i + 15
        met_min_epochs = self.epoch_progress.current.processed >= self.min_epochs if self.min_epochs else True
        met_min_steps = self.epoch_loop.global_step >= self.min_steps if self.min_steps else True
        return met_min_epochs and met_min_steps

    @property
    def _should_reload_train_dl(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if train dataloader should be reloaded.'
        n_epochs = self.trainer.reload_dataloaders_every_n_epochs
        return n_epochs and self.trainer.current_epoch - self._last_train_dl_reload_epoch >= n_epochs

    @property
    def done(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Evaluates when to leave the loop.'
        if self.max_batches == 0:
            rank_zero_info('`Trainer.fit` stopped: No training batches.')
            return True
        stop_steps = _is_max_limit_reached(self.epoch_loop.global_step, self.max_steps)
        if stop_steps:
            rank_zero_info(f'`Trainer.fit` stopped: `max_steps={self.max_steps!r}` reached.')
            return True
        assert isinstance(self.max_epochs, int)
        stop_epochs = _is_max_limit_reached(self.epoch_progress.current.processed, self.max_epochs)
        if stop_epochs:
            self.epoch_progress.current.completed = self.epoch_progress.current.processed
            rank_zero_info(f'`Trainer.fit` stopped: `max_epochs={self.max_epochs!r}` reached.')
            return True
        if self.trainer.should_stop and self._can_stop_early:
            rank_zero_debug('`Trainer.fit` stopped: `trainer.should_stop` was set.')
            return True
        return False

    @property
    def skip(self) -> bool:
        if False:
            return 10
        'Whether we should skip the training and immediately return from the call to :meth:`run`.'
        return self.done or self.trainer.limit_train_batches == 0

    def run(self) -> None:
        if False:
            return 10
        self.setup_data()
        if self.skip:
            return
        self.reset()
        self.on_run_start()
        while not self.done:
            try:
                self.on_advance_start()
                self.advance()
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        self.on_run_end()

    def setup_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._combined_loader is not None and (not self._should_reload_train_dl):
            return
        trainer = self.trainer
        pl_module = trainer.lightning_module
        if trainer.limit_train_batches == 0 or not is_overridden('training_step', pl_module):
            return
        log.debug(f'{self.__class__.__name__}: resetting train dataloader')
        source = self._data_source
        train_dataloader = _request_dataloader(source)
        trainer.strategy.barrier('train_dataloader()')
        if not isinstance(train_dataloader, CombinedLoader):
            combined_loader = CombinedLoader(train_dataloader, 'max_size_cycle')
        else:
            combined_loader = train_dataloader
        if trainer.overfit_batches > 0:
            _resolve_overfit_batches(combined_loader, mode=RunningStage.TRAINING)
        trainer_fn = TrainerFn.FITTING
        stage = RunningStage.TRAINING
        dataloaders = []
        for dl in combined_loader.flattened:
            _check_dataloader_iterable(dl, source, trainer_fn)
            dl = _process_dataloader(trainer, trainer_fn, stage, dl)
            dataloaders.append(dl)
        combined_loader.flattened = dataloaders
        self._combined_loader = combined_loader
        allow_zero_length = pl_module.allow_zero_length_dataloader_with_multiple_devices
        if trainer.datamodule is not None:
            allow_zero_length |= trainer.datamodule.allow_zero_length_dataloader_with_multiple_devices
        limits = []
        for dl in combined_loader.flattened:
            length = len(dl) if has_len_all_ranks(dl, trainer.strategy, allow_zero_length) else float('inf')
            num_batches = _parse_num_batches(stage, length, trainer.limit_train_batches)
            limits.append(num_batches)
        combined_loader.limits = limits
        self._data_fetcher = _select_data_fetcher(trainer, RunningStage.TRAINING)
        self._data_fetcher.setup(combined_loader)
        iter(self._data_fetcher)
        max_batches = sized_len(combined_loader)
        self.max_batches = max_batches if max_batches is not None else float('inf')
        has_len_all_ranks_ = has_len_all_ranks(combined_loader, trainer.strategy, allow_zero_length)
        if self.max_batches == 0:
            return
        self._last_train_dl_reload_epoch = trainer.current_epoch
        if isinstance(trainer.val_check_interval, int):
            trainer.val_check_batch = trainer.val_check_interval
            if trainer.val_check_batch > self.max_batches and trainer.check_val_every_n_epoch is not None:
                raise ValueError(f' `val_check_interval` ({trainer.val_check_interval}) must be less than or equal to the number of the training batches ({self.max_batches}). If you want to disable validation set `limit_val_batches` to 0.0 instead. If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`.')
        elif not has_len_all_ranks_:
            if trainer.val_check_interval == 1.0:
                trainer.val_check_batch = float('inf')
            else:
                raise MisconfigurationException('When using an IterableDataset for `train_dataloader`, `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies checking validation every k training batches.')
        else:
            trainer.val_check_batch = int(self.max_batches * trainer.val_check_interval)
            trainer.val_check_batch = max(1, trainer.val_check_batch)
        if trainer.loggers and self.max_batches < trainer.log_every_n_steps and (not trainer.fast_dev_run):
            rank_zero_warn(f'The number of training batches ({self.max_batches}) is smaller than the logging interval Trainer(log_every_n_steps={trainer.log_every_n_steps}). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.', category=PossibleUserWarning)

    def reset(self) -> None:
        if False:
            return 10
        'Resets the internal state of this loop.'
        assert self.trainer.model is not None
        self.trainer.model.train()
        torch.set_grad_enabled(True)
        if self.restarting:
            self.epoch_progress.reset_on_restart()

    def on_run_start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Calls the ``on_train_start`` hook.'
        if not self._iteration_based_training():
            self.epoch_progress.current.completed = self.epoch_progress.current.processed
        trainer = self.trainer
        if self.epoch_loop._should_check_val_epoch() and trainer.val_dataloaders is None:
            trainer.validating = True
            self.epoch_loop.val_loop.setup_data()
            trainer.training = True
        call._call_callback_hooks(trainer, 'on_train_start')
        call._call_lightning_module_hook(trainer, 'on_train_start')
        call._call_strategy_hook(trainer, 'on_train_start')

    def on_advance_start(self) -> None:
        if False:
            print('Hello World!')
        'Prepares the dataloader for training and calls the hook ``on_train_epoch_start``'
        trainer = self.trainer
        self.setup_data()
        assert self._combined_loader is not None
        for (i, dl) in enumerate(self._combined_loader.flattened):
            _set_sampler_epoch(dl, self.epoch_progress.current.processed)
        self.epoch_progress.increment_ready()
        call._call_callback_hooks(trainer, 'on_train_epoch_start')
        call._call_lightning_module_hook(trainer, 'on_train_epoch_start')
        self.epoch_progress.increment_started()

    def advance(self) -> None:
        if False:
            while True:
                i = 10
        'Runs one whole epoch.'
        log.debug(f'{type(self).__name__}: advancing loop')
        combined_loader = self._combined_loader
        assert combined_loader is not None
        if combined_loader._mode == 'sequential':
            raise ValueError(f"""`{type(self).__name__}` does not support the `CombinedLoader(mode="sequential")` mode. The available modes are: {[m for m in _SUPPORTED_MODES if m != 'sequential']}""")
        with self.trainer.profiler.profile('run_training_epoch'):
            assert self._data_fetcher is not None
            self.epoch_loop.run(self._data_fetcher)

    def on_advance_end(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        trainer = self.trainer
        trainer._logger_connector.epoch_end_reached()
        self.epoch_progress.increment_processed()
        call._call_callback_hooks(trainer, 'on_train_epoch_end', monitoring_callbacks=False)
        call._call_lightning_module_hook(trainer, 'on_train_epoch_end')
        call._call_callback_hooks(trainer, 'on_train_epoch_end', monitoring_callbacks=True)
        trainer._logger_connector.on_epoch_end()
        if self.epoch_loop._num_ready_batches_reached():
            self.epoch_loop.update_lr_schedulers('epoch', update_plateau_schedulers=not self.restarting)
        self.epoch_loop._batches_that_stepped -= 1
        trainer._logger_connector.update_train_epoch_metrics()
        self.epoch_loop._batches_that_stepped += 1
        self.epoch_progress.increment_completed()
        if trainer.received_sigterm:
            raise SIGTERMException

    def on_run_end(self) -> None:
        if False:
            print('Hello World!')
        'Calls the ``on_train_end`` hook.'
        log.debug(f'{self.__class__.__name__}: train run ended')
        trainer = self.trainer
        call._call_callback_hooks(trainer, 'on_train_end')
        call._call_lightning_module_hook(trainer, 'on_train_end')
        call._call_strategy_hook(trainer, 'on_train_end')

    def teardown(self) -> None:
        if False:
            while True:
                i = 10
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None
        self.epoch_loop.teardown()

    def _should_accumulate(self) -> bool:
        if False:
            print('Hello World!')
        'Whether the gradients should be accumulated.'
        return self.epoch_loop._should_accumulate()

    def _iteration_based_training(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.trainer.max_steps != -1