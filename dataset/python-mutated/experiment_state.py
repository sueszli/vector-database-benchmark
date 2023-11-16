from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import click
import logging
import os
import time
import warnings
from ray.train._internal.storage import StorageContext, get_fs_and_path, _download_from_fs_path, _list_at_fs_path
from ray.tune.experiment import Trial
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
logger = logging.getLogger(__name__)
VALID_RESUME_TYPES = [True, 'LOCAL', 'REMOTE', 'PROMPT', 'ERRORED_ONLY', 'AUTO']
_EXPERIMENT_SYNC_TIMEOUT_MESSAGE = 'If this warning keeps showing up, consider diagnosing the reason behind the hanging sync operation, or increase the `sync_timeout` in `SyncConfig`.'
_DRIVER_SYNC_EXCLUDE_PATTERNS = ['*/checkpoint_*']

@dataclass
class _ResumeConfig:
    resume_unfinished: bool = True
    resume_errored: bool = False
    restart_errored: bool = False

def _resume_str_to_config(resume_str: str) -> Tuple[str, _ResumeConfig]:
    if False:
        i = 10
        return i + 15
    if resume_str is True:
        resume_str = 'LOCAL'
    elif resume_str == 'ERRORED_ONLY':
        warnings.warn("Passing `resume='ERRORED_ONLY'` to tune.run() is deprecated and will be removed in the future. Please pass e.g. `resume='LOCAL+RESTART_ERRORED_ONLY'` instead.")
        resume_str = 'LOCAL+RESTART_ERRORED_ONLY'
    resume_config = _ResumeConfig()
    resume_settings = resume_str.split('+')
    resume_str = resume_settings[0]
    for setting in resume_settings:
        if setting == 'ERRORED':
            resume_config.resume_errored = True
        elif setting == 'RESTART_ERRORED':
            resume_config.restart_errored = True
        elif setting == 'ERRORED_ONLY':
            resume_config.resume_unfinished = False
            resume_config.restart_errored = False
            resume_config.resume_errored = True
        elif setting == 'RESTART_ERRORED_ONLY':
            resume_config.resume_unfinished = False
            resume_config.restart_errored = True
            resume_config.resume_errored = False
    assert resume_str in VALID_RESUME_TYPES, 'resume={} is not one of {}'.format(resume_str, VALID_RESUME_TYPES)
    return (resume_str, resume_config)

def _experiment_checkpoint_exists(experiment_dir: str) -> bool:
    if False:
        return 10
    return bool(_find_newest_experiment_checkpoint(experiment_dir=experiment_dir))

def _find_newest_experiment_checkpoint(experiment_dir: str) -> Optional[str]:
    if False:
        print('Hello World!')
    'Returns file name of most recently created experiment checkpoint.\n\n    Args:\n        experiment_dir: Local or remote path to the experiment directory\n            containing at least one experiment checkpoint file.\n\n    Returns:\n        str: The local or remote path to the latest experiment checkpoint file\n            based on timestamp. None if no experiment checkpoints were found.\n    '
    from ray.tune.analysis import ExperimentAnalysis
    (fs, path) = get_fs_and_path(experiment_dir)
    return ExperimentAnalysis._find_newest_experiment_checkpoint(fs=fs, experiment_fs_path=path)

class _ExperimentCheckpointManager:
    """Helper class for managing experiment-level checkpoints.

    This class implements the ``checkpoint()`` method used to checkpoint
    experiment state. When called, this will serialize and write to disk
    the state of the trial runner, trial executor, and search algorithm, to
    a specified checkpoint file.

    The checkpoint period is automatically adjusted to
    ``max(10, time_per_checkpoint * 19)``. This means that at most 5% of the
    time (1/20) will be used for writing checkpoints, while 95% of the time
    (19/20) will be used to handle the rest of the training loop.

    If ``sync_every_n_trial_checkpoints`` is not None, syncing
    to cloud will be forced if any trial has checkpointed more times than
    ``sync_every_n_trial_checkpoints`` since last sync.

    """

    def __init__(self, *, storage: Optional[StorageContext], checkpoint_period: Union[int, float, str], sync_every_n_trial_checkpoints: Optional[int]=None):
        if False:
            return 10
        self._storage = storage
        self._last_save_time = 0.0
        self._last_sync_time = 0.0
        self._auto_checkpoint_enabled = checkpoint_period == 'auto'
        if self._auto_checkpoint_enabled:
            self._checkpoint_period = 10.0
        else:
            self._checkpoint_period = float(checkpoint_period)
        self._sync_every_n_trial_checkpoints = sync_every_n_trial_checkpoints
        self._trial_num_checkpoints_since_last_sync: Dict[Trial, int] = Counter()
        self._slow_sync_threshold = float(os.environ.get('TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S', '30'))
        self._excessive_sync_threshold = float(os.environ.get('TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S', '30'))
        self._should_force_cloud_sync = False

    @property
    def auto_checkpoint_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return self._auto_checkpoint_enabled

    def _update_auto_checkpoint_time(self, time_taken: float):
        if False:
            print('Hello World!')
        if self._auto_checkpoint_enabled:
            self._checkpoint_period = max(10.0, time_taken * 19)
            logger.debug(f'Global experiment checkpointing took {time_taken:.2f} seconds. Adjusting checkpoint period to {self._checkpoint_period:.2f} seconds.')

    def on_trial_checkpoint(self, trial: Trial):
        if False:
            print('Hello World!')
        if not self._sync_every_n_trial_checkpoints:
            return
        self._trial_num_checkpoints_since_last_sync[trial] += 1
        if self._trial_num_checkpoints_since_last_sync[trial] >= self._sync_every_n_trial_checkpoints:
            self._should_force_cloud_sync = True

    def checkpoint(self, save_fn: Callable[[], None], force: bool=False, wait: bool=False):
        if False:
            i = 10
            return i + 15
        'Saves execution state to the local experiment directory.\n        Overwrites the current session checkpoint, which starts when self\n        is instantiated. Throttle depends on self._checkpoint_period.\n\n        Also, automatically saves the search algorithm to the local\n        checkpoint dir.\n\n        Args:\n            save_fn: Function to call to actually save data. Should expect\n                one string argument specifying the directory to save to.\n            force: Forces a checkpoint despite checkpoint_period.\n            wait: Wait until sync to cloud has finished.\n\n        '
        experiment_local_path = self._storage.experiment_local_path
        if not experiment_local_path:
            return
        force = force or self._should_force_cloud_sync
        now = time.time()
        if now - self._last_save_time < self._checkpoint_period and (not force):
            return
        checkpoint_time_start = time.monotonic()
        with out_of_band_serialize_dataset():
            save_fn()
        self.sync_up(force=force, wait=wait)
        checkpoint_time_taken = time.monotonic() - checkpoint_time_start
        self._update_auto_checkpoint_time(time_taken=checkpoint_time_taken)
        self._last_save_time = time.time()
        return experiment_local_path

    def sync_up(self, force: bool=False, wait: bool=False) -> bool:
        if False:
            return 10
        syncer = self._storage.syncer
        if not syncer:
            return False
        exclude = _DRIVER_SYNC_EXCLUDE_PATTERNS
        experiment_local_path = self._storage.experiment_local_path
        experiment_fs_path = self._storage.experiment_fs_path
        if force:
            try:
                syncer.wait()
            except TimeoutError as e:
                logger.warning(f'The previous sync of the experiment directory to the cloud timed out with the error: {str(e)}\nSyncing will be retried. ' + _EXPERIMENT_SYNC_TIMEOUT_MESSAGE)
            except Exception as e:
                logger.warning(f'The previous sync of the experiment directory to the cloud failed with the error: {str(e)}\nSyncing will be retried.')
            synced = syncer.sync_up(local_dir=experiment_local_path, remote_dir=experiment_fs_path, exclude=exclude)
        else:
            synced = syncer.sync_up_if_needed(local_dir=experiment_local_path, remote_dir=experiment_fs_path, exclude=exclude)
        start_time = time.monotonic()
        if wait:
            try:
                syncer.wait()
            except Exception as e:
                raise RuntimeError(f'Uploading the experiment directory from the driver (local path: {experiment_local_path}) to the the cloud (remote path: {experiment_fs_path}) failed. Please check the error message above.') from e
        now = time.monotonic()
        sync_time_taken = now - start_time
        if sync_time_taken > self._slow_sync_threshold:
            try:
                import fsspec
            except Exception:
                fsspec = None
            fsspec_msg = ''
            if fsspec is None:
                fsspec_msg = 'If your data is small, try installing fsspec (`pip install fsspec`) for more efficient local file parsing. '
            logger.warning(f'Syncing the experiment checkpoint to cloud took a long time with {sync_time_taken:.2f} seconds. This can be due to a large number of trials, large logfiles, or throttling from the remote storage provider for too frequent syncs. {fsspec_msg}If your `CheckpointConfig.num_to_keep` is a low number, this can trigger frequent syncing, in which case you should increase it. ')
        if not synced:
            return False
        self._should_force_cloud_sync = False
        self._trial_num_checkpoints_since_last_sync.clear()
        if now - self._last_sync_time < self._excessive_sync_threshold:
            logger.warning(f'Experiment checkpoint syncing has been triggered multiple times in the last {self._excessive_sync_threshold} seconds. A sync will be triggered whenever a trial has checkpointed more than `num_to_keep` times since last sync or if {syncer.sync_period} seconds have passed since last sync. If you have set `num_to_keep` in your `CheckpointConfig`, consider increasing the checkpoint frequency or keeping more checkpoints. You can supress this warning by changing the `TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S` environment variable.')
        self._last_sync_time = now
        return True

    def sync_down_experiment_state(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        fs = self._storage.storage_filesystem
        filepaths = _list_at_fs_path(fs=fs, fs_path=self._storage.experiment_fs_path)
        matches = [path for path in filepaths if path.endswith('.json') or path.endswith('.pkl')]
        for relpath in matches:
            fs_path = Path(self._storage.experiment_fs_path, relpath).as_posix()
            local_path = Path(self._storage.experiment_local_path, relpath).as_posix()
            _download_from_fs_path(fs=fs, fs_path=fs_path, local_path=local_path)
        logger.debug(f'Copied {matches} from:\n(fs, path) = ({self._storage.storage_filesystem.type_name}, {self._storage.experiment_fs_path})\n-> {self._storage.experiment_local_path}')

    def _resume_auto(self) -> bool:
        if False:
            return 10
        experiment_local_path = self._storage.experiment_local_path
        experiment_fs_path = self._storage.experiment_fs_path
        syncer = self._storage.syncer
        if experiment_fs_path and syncer:
            logger.info(f'Trying to find and download experiment checkpoint at {experiment_fs_path}')
            try:
                self.sync_down_experiment_state()
            except Exception:
                logger.exception("Got error when trying to sync down.\nPlease check this error message for potential access problems - if a directory was not found, that is expected at this stage when you're starting a new experiment.")
                logger.info('No remote checkpoint was found or an error occurred when trying to download the experiment checkpoint. Please check the previous warning message for more details. Starting a new run...')
                return False
            if not _experiment_checkpoint_exists(experiment_local_path):
                logger.warning('A remote checkpoint was fetched, but no checkpoint data was found. This can happen when e.g. the cloud bucket exists but does not contain any data. Starting a new run...')
                return False
            logger.info('A remote experiment checkpoint was found and will be used to restore the previous experiment state.')
            return True
        elif not _experiment_checkpoint_exists(experiment_local_path):
            logger.info('No local checkpoint was found. Starting a new run...')
            return False
        logger.info('A local experiment checkpoint was found and will be used to restore the previous experiment state.')
        return True

    def resume(self, resume_type: Union[str, bool]) -> Optional[_ResumeConfig]:
        if False:
            while True:
                i = 10
        'Checks whether to resume experiment.\n\n        If experiment should be resumed, this method may sync down experiment state\n        from the cloud and then return a ResumeConfig mapping to the resume type.\n\n        Args:\n            resume_type: One of ["REMOTE", "LOCAL", "PROMPT", "AUTO"]. Can\n                be suffixed with one or more of ["+ERRORED", "+ERRORED_ONLY",\n                "+RESTART_ERRORED", "+RESTART_ERRORED_ONLY"]\n\n        Returns:\n            _ResumeConfig if resume is successful. None otherwise.\n        '
        if not resume_type:
            return None
        (resume_type, resume_config) = _resume_str_to_config(resume_type)
        experiment_local_path = self._storage.experiment_local_path
        experiment_fs_path = self._storage.experiment_fs_path
        if resume_type == 'AUTO':
            if self._resume_auto():
                return resume_config
            return None
        if resume_type in ['LOCAL', 'PROMPT']:
            if not _experiment_checkpoint_exists(experiment_local_path):
                raise ValueError(f'You called resume ({resume_type}) when no checkpoint exists in local directory ({experiment_local_path}). If you want to start a new experiment, use `resume="AUTO"` or `resume=None`. If you expected an experiment to already exist, check if you supplied the correct `local_dir` to `train.RunConfig()`.')
            elif resume_type == 'PROMPT':
                if click.confirm(f'Resume from local directory? ({experiment_local_path})'):
                    return resume_config
        if resume_type in ['REMOTE', 'PROMPT']:
            if resume_type == 'PROMPT' and (not click.confirm(f'Try downloading from remote directory? ({experiment_fs_path})')):
                return None
            logger.info(f'Downloading experiment checkpoint from {experiment_fs_path}')
            self.sync_down_experiment_state()
            if not _experiment_checkpoint_exists(experiment_local_path):
                raise ValueError('Called resume when no checkpoint exists in remote or local directory.')
        return resume_config