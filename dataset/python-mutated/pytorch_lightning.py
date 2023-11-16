import warnings
from packaging import version
import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
_EPOCH_KEY = 'ddp_pl:epoch'
_INTERMEDIATE_VALUE = 'ddp_pl:intermediate_value'
_PRUNED_KEY = 'ddp_pl:pruned'
with optuna._imports.try_import() as _imports:
    import lightning.pytorch as pl
    from lightning.pytorch import LightningModule
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import Callback
if not _imports.is_successful():
    Callback = object
    LightningModule = object
    Trainer = object

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``lightning.pytorch.LightningModule.training_step`` or
            ``lightning.pytorch.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.


    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.6.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.


    .. note::
        If you would like to use PyTorchLightningPruningCallback in a distributed training
        environment, you need to evoke `PyTorchLightningPruningCallback.check_pruned()`
        manually so that :class:`~optuna.exceptions.TrialPruned` is properly handled.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        if False:
            i = 10
            return i + 15
        _imports.check()
        super().__init__()
        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer: Trainer, pl_module: 'pl.LightningModule') -> None:
        if False:
            print('Hello World!')
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if self.is_ddp_backend:
            if version.parse(pl.__version__) < version.parse('1.6.0'):
                raise ValueError('PyTorch Lightning>=1.6.0 is required in DDP.')
            if not (isinstance(self._trial.study._storage, _CachedStorage) and isinstance(self._trial.study._storage._backend, RDBStorage)):
                raise ValueError('optuna.integration.PyTorchLightningPruningCallback supports only optuna.storages.RDBStorage in DDP.')
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(self._trial._trial_id, _INTERMEDIATE_VALUE, dict())

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if False:
            for i in range(10):
                print('nop')
        if trainer.sanity_checking:
            return
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = f"The metric '{self.monitor}' is not in the evaluation logs for pruning. Please make sure you set the correct metric name."
            warnings.warn(message)
            return
        epoch = pl_module.current_epoch
        should_stop = False
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f'Trial was pruned at epoch {epoch}.')
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values)
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return
        if trainer.is_global_zero:
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _PRUNED_KEY, True)
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _EPOCH_KEY, epoch)

    def check_pruned(self) -> None:
        if False:
            while True:
                i = 10
        "Raise :class:`optuna.TrialPruned` manually if pruned.\n\n        Currently, ``intermediate_values`` are not properly propagated between processes due to\n        storage cache. Therefore, necessary information is kept in trial_system_attrs when the\n        trial runs in a distributed situation. Please call this method right after calling\n        ``lightning.pytorch.Trainer.fit()``.\n        If a callback doesn't have any backend storage for DDP, this method does nothing.\n        "
        _trial_id = self._trial._trial_id
        _study = self._trial.study
        if not isinstance(_study._storage, _CachedStorage):
            return
        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
        if intermediate_values is None:
            return
        for (epoch, score) in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY)
            raise optuna.TrialPruned(f'Trial was pruned at epoch {epoch}.')