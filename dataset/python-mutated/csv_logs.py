"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from lightning.fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning.fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.logger import _convert_params
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
log = logging.getLogger(__name__)

class ExperimentWriter(_FabricExperimentWriter):
    """Experiment writer for CSVLogger.

    Currently, supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    This logger supports logging to remote filesystems via ``fsspec``. Make sure you have it installed.

    Args:
        log_dir: Directory for the experiment logs

    """
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(self, log_dir: str) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(log_dir=log_dir)
        self.hparams: Dict[str, Any] = {}

    def log_hparams(self, params: Dict[str, Any]) -> None:
        if False:
            return 10
        'Record hparams.'
        self.hparams.update(params)

    @override
    def save(self) -> None:
        if False:
            i = 10
            return i + 15
        'Save recorded hparams and metrics into files.'
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)
        return super().save()

class CSVLogger(Logger, FabricCSVLogger):
    """Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'lightning_logs'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    """
    LOGGER_JOIN_CHAR = '-'

    def __init__(self, save_dir: _PATH, name: str='lightning_logs', version: Optional[Union[int, str]]=None, prefix: str='', flush_logs_every_n_steps: int=100):
        if False:
            i = 10
            return i + 15
        super().__init__(root_dir=save_dir, name=name, version=version, prefix=prefix, flush_logs_every_n_steps=flush_logs_every_n_steps)
        self._save_dir = os.fspath(save_dir)

    @property
    @override
    def root_dir(self) -> str:
        if False:
            i = 10
            return i + 15
        'Parent directory for all checkpoint subdirectories.\n\n        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will\n        be saved in "save_dir/version"\n\n        '
        return os.path.join(self.save_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        if False:
            i = 10
            return i + 15
        "The log directory for this run.\n\n        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the\n        constructor's version parameter instead of ``None`` or an int.\n\n        "
        version = self.version if isinstance(self.version, str) else f'version_{self.version}'
        return os.path.join(self.root_dir, version)

    @property
    @override
    def save_dir(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The current directory where logs are saved.\n\n        Returns:\n            The path to current directory where logs are saved.\n\n        '
        return self._save_dir

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if False:
            print('Hello World!')
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @property
    @override
    @rank_zero_experiment
    def experiment(self) -> _FabricExperimentWriter:
        if False:
            print('Hello World!')
        'Actual _ExperimentWriter object. To use _ExperimentWriter features in your\n        :class:`~lightning.pytorch.core.LightningModule` do the following.\n\n        Example::\n\n            self.logger.experiment.some_experiment_writer_function()\n\n        '
        if self._experiment is not None:
            return self._experiment
        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment