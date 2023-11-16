import logging
import os
import re
from typing import Any, Dict, Optional
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from torch import Tensor
import lightning.pytorch as pl
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
log = logging.getLogger(__name__)

class _CheckpointConnector:

    def __init__(self, trainer: 'pl.Trainer') -> None:
        if False:
            i = 10
            return i + 15
        self.trainer = trainer
        self._ckpt_path: Optional[_PATH] = None
        self._user_managed: bool = False
        self._loaded_checkpoint: Dict[str, Any] = {}

    @property
    def _hpc_resume_path(self) -> Optional[str]:
        if False:
            print('Hello World!')
        dir_path_hpc = self.trainer.default_root_dir
        dir_path_hpc = str(dir_path_hpc)
        (fs, path) = url_to_fs(dir_path_hpc)
        if not _is_dir(fs, path):
            return None
        max_version = self.__max_ckpt_version_in_folder(dir_path_hpc, 'hpc_ckpt_')
        if max_version is not None:
            if isinstance(fs, LocalFileSystem):
                return os.path.join(dir_path_hpc, f'hpc_ckpt_{max_version}.ckpt')
            return dir_path_hpc + fs.sep + f'hpc_ckpt_{max_version}.ckpt'
        return None

    def resume_start(self, checkpoint_path: Optional[_PATH]=None) -> None:
        if False:
            while True:
                i = 10
        'Attempts to pre-load the checkpoint file to memory, with the source path determined in this priority:\n\n        1. from HPC weights if `checkpoint_path` is ``None`` and on SLURM or passed keyword `"hpc"`.\n        2. from fault-tolerant auto-saved checkpoint if found\n        3. from `checkpoint_path` file if provided\n        4. don\'t restore\n\n        '
        self._ckpt_path = checkpoint_path
        if not checkpoint_path:
            log.debug('`checkpoint_path` not specified. Skipping checkpoint loading.')
            return
        rank_zero_info(f'Restoring states from the checkpoint path at {checkpoint_path}')
        with pl_legacy_patch():
            loaded_checkpoint = self.trainer.strategy.load_checkpoint(checkpoint_path)
        self._loaded_checkpoint = _pl_migrate_checkpoint(loaded_checkpoint, checkpoint_path)

    def _select_ckpt_path(self, state_fn: TrainerFn, ckpt_path: Optional[_PATH], model_provided: bool, model_connected: bool) -> Optional[_PATH]:
        if False:
            print('Hello World!')
        'Called by the ``Trainer`` to select the checkpoint path source.'
        if self._user_managed:
            if ckpt_path:
                rank_zero_warn(f'`trainer.ckpt_path = {self._ckpt_path!r}` was called but then you passed `trainer.fit(ckpt_path={ckpt_path!r})`. The latter will be loaded.')
                self._ckpt_path = None
                self._user_managed = False
                ckpt_path = self._parse_ckpt_path(state_fn, ckpt_path, model_provided=model_provided, model_connected=model_connected)
            else:
                ckpt_path = self._ckpt_path
        else:
            ckpt_path = self._parse_ckpt_path(state_fn, ckpt_path, model_provided=model_provided, model_connected=model_connected)
        return ckpt_path

    def _parse_ckpt_path(self, state_fn: TrainerFn, ckpt_path: Optional[_PATH], model_provided: bool, model_connected: bool) -> Optional[_PATH]:
        if False:
            while True:
                i = 10
        'Converts the ``ckpt_path`` special values into an actual filepath, depending on the trainer\n        configuration.'
        if ckpt_path is None and SLURMEnvironment.detect() and (self._hpc_resume_path is not None):
            ckpt_path = 'hpc'
        from lightning.pytorch.callbacks.on_exception_checkpoint import OnExceptionCheckpoint
        ft_checkpoints = [cb for cb in self.trainer.callbacks if isinstance(cb, OnExceptionCheckpoint)]
        fn = state_fn.value
        if ckpt_path is None and ft_checkpoints and (self.trainer.state.fn == TrainerFn.FITTING):
            ckpt_path = 'last'
            rank_zero_warn(f"`.{fn}(ckpt_path=None)` was called without a model. The last model of the previous `fit` call will be used. You can pass `{fn}(ckpt_path='best')` to use the best model or `{fn}(ckpt_path='last')` to use the last model. If you pass a value, this warning will be silenced.")
        if model_provided and ckpt_path is None:
            return None
        if model_connected and ckpt_path is None:
            ckpt_path = 'best'
            ft_tip = ' There is also an on-exception checkpoint available, however it is used by default only when fitting.' if ft_checkpoints else ''
            rank_zero_warn(f'`.{fn}(ckpt_path=None)` was called without a model. The best model of the previous `fit` call will be used.' + ft_tip + f" You can pass `.{fn}(ckpt_path='best')` to use the best model or `.{fn}(ckpt_path='last')` to use the last model. If you pass a value, this warning will be silenced.")
        if ckpt_path == 'best':
            if len(self.trainer.checkpoint_callbacks) > 1:
                rank_zero_warn(f'`.{fn}(ckpt_path="best")` is called with Trainer configured with multiple `ModelCheckpoint` callbacks. It will use the best checkpoint path from first checkpoint callback.')
            if not self.trainer.checkpoint_callback:
                raise ValueError(f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured.')
            has_best_model_path = self.trainer.checkpoint_callback.best_model_path
            if hasattr(self.trainer.checkpoint_callback, 'best_model_path') and (not has_best_model_path):
                if self.trainer.fast_dev_run:
                    raise ValueError(f'You cannot execute `.{fn}(ckpt_path="best")` with `fast_dev_run=True`. Please pass an exact checkpoint path to `.{fn}(ckpt_path=...)`')
                raise ValueError(f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.')
            ckpt_path = getattr(self.trainer.checkpoint_callback, 'best_model_path', None)
        elif ckpt_path == 'last':
            candidates = {getattr(ft, 'ckpt_path', None) for ft in ft_checkpoints}
            for callback in self.trainer.checkpoint_callbacks:
                if isinstance(callback, ModelCheckpoint):
                    candidates |= callback._find_last_checkpoints(self.trainer)
            candidates_fs = {path: get_filesystem(path) for path in candidates if path}
            candidates_ts = {path: fs.modified(path) for (path, fs) in candidates_fs.items() if fs.exists(path)}
            if not candidates_ts:
                rank_zero_warn(f'.{fn}(ckpt_path="last") is set, but there is no last checkpoint available. No checkpoint will be loaded.')
                return None
            ckpt_path = max(candidates_ts, key=candidates_ts.get)
        elif ckpt_path == 'hpc':
            if not self._hpc_resume_path:
                raise ValueError(f'`.{fn}(ckpt_path="hpc")` is set but no HPC checkpoint was found. Please pass an exact checkpoint path to `.{{fn}}(ckpt_path=...)`')
            ckpt_path = self._hpc_resume_path
        if not ckpt_path:
            raise ValueError(f'`.{fn}()` found no path for the best weights: {ckpt_path!r}. Please specify a path for a checkpoint `.{fn}(ckpt_path=PATH)`')
        return ckpt_path

    def resume_end(self) -> None:
        if False:
            i = 10
            return i + 15
        'Signal the connector that all states have resumed and memory for the checkpoint object can be released.'
        assert self.trainer.state.fn is not None
        if self._ckpt_path:
            message = 'Restored all states' if self.trainer.state.fn == TrainerFn.FITTING else 'Loaded model weights'
            rank_zero_info(f'{message} from the checkpoint at {self._ckpt_path}')
        self._loaded_checkpoint = {}
        torch.cuda.empty_cache()
        self.trainer.strategy.barrier('_CheckpointConnector.resume_end')

    def restore(self, checkpoint_path: Optional[_PATH]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Attempt to restore everything at once from a 'PyTorch-Lightning checkpoint' file through file-read and\n        state-restore, in this priority:\n\n        1. from HPC weights if found\n        2. from `checkpoint_path` file if provided\n        3. don't restore\n\n        All restored states are listed in return value description of `dump_checkpoint`.\n\n        Args:\n            checkpoint_path: Path to a PyTorch Lightning checkpoint file.\n\n        "
        self.resume_start(checkpoint_path)
        self.restore_datamodule()
        self.restore_model()
        self.restore_callbacks()
        self.restore_training_state()
        self.resume_end()

    def restore_datamodule(self) -> None:
        if False:
            i = 10
            return i + 15
        'Calls hooks on the datamodule to give it a chance to restore its state from the checkpoint.'
        if not self._loaded_checkpoint:
            return
        trainer = self.trainer
        datamodule = trainer.datamodule
        if datamodule is not None and datamodule.__class__.__qualname__ in self._loaded_checkpoint:
            call._call_lightning_datamodule_hook(trainer, 'load_state_dict', self._loaded_checkpoint[datamodule.__class__.__qualname__])

    def restore_model(self) -> None:
        if False:
            while True:
                i = 10
        "Restores a model's weights from a PyTorch Lightning checkpoint.\n\n        Hooks are called first to give the LightningModule a chance to modify the contents, then finally the model gets\n        updated with the loaded weights.\n\n        "
        if not self._loaded_checkpoint:
            return
        trainer = self.trainer
        call._call_lightning_module_hook(trainer, 'on_load_checkpoint', self._loaded_checkpoint)
        trainer.strategy.load_model_state_dict(self._loaded_checkpoint)

    def restore_training_state(self) -> None:
        if False:
            print('Hello World!')
        'Restore the trainer state from the pre-loaded checkpoint.\n\n        This includes the precision settings, loop progress, optimizer states and learning rate scheduler states.\n\n        '
        if not self._loaded_checkpoint:
            return
        self.restore_precision_plugin_state()
        self.restore_loops()
        assert self.trainer.state.fn is not None
        if self.trainer.state.fn == TrainerFn.FITTING:
            self.restore_optimizers_and_schedulers()

    def restore_precision_plugin_state(self) -> None:
        if False:
            print('Hello World!')
        'Restore the precision plugin state from the pre-loaded checkpoint.'
        prec_plugin = self.trainer.precision_plugin
        prec_plugin.on_load_checkpoint(self._loaded_checkpoint)
        if prec_plugin.__class__.__qualname__ in self._loaded_checkpoint:
            prec_plugin.load_state_dict(self._loaded_checkpoint[prec_plugin.__class__.__qualname__])
        if 'native_amp_scaling_state' in self._loaded_checkpoint and isinstance(prec_plugin, MixedPrecision):
            prec_plugin.load_state_dict(self._loaded_checkpoint['native_amp_scaling_state'])

    def restore_callbacks(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Restores all callbacks from the pre-loaded checkpoint.'
        if not self._loaded_checkpoint:
            return
        trainer = self.trainer
        call._call_callbacks_on_load_checkpoint(trainer, self._loaded_checkpoint)
        call._call_callbacks_load_state_dict(trainer, self._loaded_checkpoint)

    def restore_loops(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Restores the loop progress from the pre-loaded checkpoint.\n\n        Calls hooks on the loops to give it a chance to restore its state from the checkpoint.\n\n        '
        if not self._loaded_checkpoint:
            return
        fit_loop = self.trainer.fit_loop
        assert self.trainer.state.fn is not None
        state_dict = self._loaded_checkpoint.get('loops')
        if state_dict is not None:
            if self.trainer.state.fn == TrainerFn.FITTING:
                fit_loop.load_state_dict(state_dict['fit_loop'])
            elif self.trainer.state.fn == TrainerFn.VALIDATING:
                self.trainer.validate_loop.load_state_dict(state_dict['validate_loop'])
            elif self.trainer.state.fn == TrainerFn.TESTING:
                self.trainer.test_loop.load_state_dict(state_dict['test_loop'])
            elif self.trainer.state.fn == TrainerFn.PREDICTING:
                self.trainer.predict_loop.load_state_dict(state_dict['predict_loop'])
        if self.trainer.state.fn != TrainerFn.FITTING:
            return
        if self.trainer.max_epochs != -1 and self.trainer.max_epochs is not None and (self.trainer.current_epoch > self.trainer.max_epochs):
            raise MisconfigurationException(f'You restored a checkpoint with current_epoch={self.trainer.current_epoch}, but you have set Trainer(max_epochs={self.trainer.max_epochs}).')

    def restore_optimizers_and_schedulers(self) -> None:
        if False:
            return 10
        'Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint.'
        if not self._loaded_checkpoint:
            return
        if self.trainer.strategy.lightning_restore_optimizer:
            if 'optimizer_states' not in self._loaded_checkpoint:
                raise KeyError('Trying to restore optimizer state but checkpoint contains only the model. This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.')
            self.restore_optimizers()
        if 'lr_schedulers' not in self._loaded_checkpoint:
            raise KeyError('Trying to restore learning rate scheduler state but checkpoint contains only the model. This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.')
        self.restore_lr_schedulers()

    def restore_optimizers(self) -> None:
        if False:
            while True:
                i = 10
        'Restores the optimizer states from the pre-loaded checkpoint.'
        if not self._loaded_checkpoint:
            return
        self.trainer.strategy.load_optimizer_state_dict(self._loaded_checkpoint)

    def restore_lr_schedulers(self) -> None:
        if False:
            print('Hello World!')
        'Restores the learning rate scheduler states from the pre-loaded checkpoint.'
        if not self._loaded_checkpoint:
            return
        lr_schedulers = self._loaded_checkpoint['lr_schedulers']
        for (config, lrs_state) in zip(self.trainer.lr_scheduler_configs, lr_schedulers):
            config.scheduler.load_state_dict(lrs_state)

    def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.resume_start(checkpoint_path)
        self.restore_model()
        self.restore_datamodule()
        if self.trainer.state.fn == TrainerFn.FITTING:
            self.restore_callbacks()

    def dump_checkpoint(self, weights_only: bool=False) -> dict:
        if False:
            while True:
                i = 10
        'Creating a model checkpoint dictionary object from various component states.\n\n        Args:\n            weights_only: saving model weights only\n        Return:\n            structured dictionary: {\n                \'epoch\':                     training epoch\n                \'global_step\':               training global step\n                \'pytorch-lightning_version\': The version of PyTorch Lightning that produced this checkpoint\n                \'callbacks\':                 "callback specific state"[] # if not weights_only\n                \'optimizer_states\':          "PT optim\'s state_dict"[]   # if not weights_only\n                \'lr_schedulers\':             "PT sched\'s state_dict"[]   # if not weights_only\n                \'state_dict\':                Model\'s state_dict (e.g. network weights)\n                precision_plugin.__class__.__qualname__:  precision plugin state_dict # if not weights_only\n                CHECKPOINT_HYPER_PARAMS_NAME:\n                CHECKPOINT_HYPER_PARAMS_KEY:\n                CHECKPOINT_HYPER_PARAMS_TYPE:\n                something_cool_i_want_to_save: anything you define through model.on_save_checkpoint\n                LightningDataModule.__class__.__qualname__: pl DataModule\'s state\n            }\n\n        '
        trainer = self.trainer
        model = trainer.lightning_module
        datamodule = trainer.datamodule
        checkpoint = {'epoch': trainer.current_epoch, 'global_step': trainer.global_step, 'pytorch-lightning_version': pl.__version__, 'state_dict': self._get_lightning_module_state_dict(), 'loops': self._get_loops_state_dict()}
        if not weights_only:
            checkpoint['callbacks'] = call._call_callbacks_state_dict(trainer)
            optimizer_states = []
            for (i, optimizer) in enumerate(trainer.optimizers):
                optimizer_state = trainer.strategy.optimizer_state(optimizer)
                optimizer_states.append(optimizer_state)
            checkpoint['optimizer_states'] = optimizer_states
            lr_schedulers = []
            for config in trainer.lr_scheduler_configs:
                lr_schedulers.append(config.scheduler.state_dict())
            checkpoint['lr_schedulers'] = lr_schedulers
            prec_plugin = trainer.precision_plugin
            prec_plugin_state_dict = prec_plugin.state_dict()
            if prec_plugin_state_dict:
                checkpoint[prec_plugin.__class__.__qualname__] = prec_plugin_state_dict
            prec_plugin.on_save_checkpoint(checkpoint)
        if _OMEGACONF_AVAILABLE:
            from omegaconf import Container
        for obj in (model, datamodule):
            if obj and obj.hparams:
                if hasattr(obj, '_hparams_name'):
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_NAME] = obj._hparams_name
                if _OMEGACONF_AVAILABLE and isinstance(obj.hparams, Container):
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = obj.hparams
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_TYPE] = type(obj.hparams)
                else:
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = dict(obj.hparams)
        if datamodule is not None:
            datamodule_state_dict = call._call_lightning_datamodule_hook(trainer, 'state_dict')
            if datamodule_state_dict:
                checkpoint[datamodule.__class__.__qualname__] = datamodule_state_dict
        if not weights_only:
            call._call_callbacks_on_save_checkpoint(trainer, checkpoint)
        call._call_lightning_module_hook(trainer, 'on_save_checkpoint', checkpoint)
        return checkpoint

    def _get_lightning_module_state_dict(self) -> Dict[str, Tensor]:
        if False:
            return 10
        return self.trainer.strategy.lightning_module_state_dict()

    def _get_loops_state_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'fit_loop': self.trainer.fit_loop.state_dict(), 'validate_loop': self.trainer.validate_loop.state_dict(), 'test_loop': self.trainer.test_loop.state_dict(), 'predict_loop': self.trainer.predict_loop.state_dict()}

    @staticmethod
    def __max_ckpt_version_in_folder(dir_path: _PATH, name_key: str='ckpt_') -> Optional[int]:
        if False:
            return 10
        'List up files in `dir_path` with `name_key`, then yield maximum suffix number.\n\n        Args:\n            dir_path: path of directory which may contain files whose name include `name_key`\n            name_key: file name prefix\n        Returns:\n            None if no-corresponding-file else maximum suffix number\n\n        '
        (fs, uri) = url_to_fs(str(dir_path))
        if not fs.exists(dir_path):
            return None
        files = [os.path.basename(f['name']) for f in fs.listdir(uri)]
        files = [x for x in files if name_key in x]
        if len(files) == 0:
            return None
        ckpt_vs = []
        for name in files:
            name = name.split(name_key)[-1]
            name = re.sub('[^0-9]', '', name)
            ckpt_vs.append(int(name))
        return max(ckpt_vs)

    @staticmethod
    def __get_max_ckpt_path_from_folder(folder_path: _PATH) -> str:
        if False:
            while True:
                i = 10
        'Get path of maximum-epoch checkpoint in the folder.'
        max_suffix = _CheckpointConnector.__max_ckpt_version_in_folder(folder_path)
        ckpt_number = max_suffix if max_suffix is not None else 0
        return f'{folder_path}/hpc_ckpt_{ckpt_number}.ckpt'

    @staticmethod
    def hpc_save_path(folderpath: _PATH) -> str:
        if False:
            print('Hello World!')
        max_suffix = _CheckpointConnector.__max_ckpt_version_in_folder(folderpath)
        ckpt_number = (max_suffix if max_suffix is not None else 0) + 1
        return os.path.join(folderpath, f'hpc_ckpt_{ckpt_number}.ckpt')