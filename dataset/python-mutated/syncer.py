import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
logger = logging.getLogger(__name__)
DEFAULT_SYNC_PERIOD = 300
DEFAULT_SYNC_TIMEOUT = 1800

@PublicAPI(stability='stable')
@dataclass
class SyncConfig:
    """Configuration object for Train/Tune file syncing to `RunConfig(storage_path)`.

    In Ray Train/Tune, here is where syncing (mainly uploading) happens:

    The experiment driver (on the head node) syncs the experiment directory to storage
    (which includes experiment state such as searcher state, the list of trials
    and their statuses, and trial metadata).

    It's also possible to sync artifacts from the trial directory to storage
    by setting `sync_artifacts=True`.
    For a Ray Tune run with many trials, each trial will upload its trial directory
    to storage, which includes arbitrary files that you dumped during the run.
    For a Ray Train run doing distributed training, each remote worker will similarly
    upload its trial directory to storage.

    See :ref:`persistent-storage-guide` for more details and examples.

    Args:
        sync_period: Minimum time in seconds to wait between two sync operations.
            A smaller ``sync_period`` will have the data in storage updated more often
            but introduces more syncing overhead. Defaults to 5 minutes.
        sync_timeout: Maximum time in seconds to wait for a sync process
            to finish running. A sync operation will run for at most this long
            before raising a `TimeoutError`. Defaults to 30 minutes.
        sync_artifacts: [Beta] Whether or not to sync artifacts that are saved to the
            trial directory (accessed via `train.get_context().get_trial_dir()`)
            to the persistent storage configured via `train.RunConfig(storage_path)`.
            The trial or remote worker will try to launch an artifact syncing
            operation every time `train.report` happens, subject to `sync_period`
            and `sync_artifacts_on_checkpoint`.
            Defaults to False -- no artifacts are persisted by default.
        sync_artifacts_on_checkpoint: If True, trial/worker artifacts are
            forcefully synced on every reported checkpoint.
            This only has an effect if `sync_artifacts` is True.
            Defaults to True.
    """
    sync_period: int = DEFAULT_SYNC_PERIOD
    sync_timeout: int = DEFAULT_SYNC_TIMEOUT
    sync_artifacts: bool = False
    sync_artifacts_on_checkpoint: bool = True
    upload_dir: Optional[str] = _DEPRECATED_VALUE
    syncer: Optional[Union[str, 'Syncer']] = _DEPRECATED_VALUE
    sync_on_checkpoint: bool = _DEPRECATED_VALUE

    def _deprecation_warning(self, attr_name: str, extra_msg: str):
        if False:
            while True:
                i = 10
        if getattr(self, attr_name) != _DEPRECATED_VALUE:
            if log_once(f'sync_config_param_deprecation_{attr_name}'):
                warnings.warn(f'`SyncConfig({attr_name})` is a deprecated configuration and will be ignored. Please remove it from your `SyncConfig`, as this will raise an error in a future version of Ray.{extra_msg}')

    def __post_init__(self):
        if False:
            print('Hello World!')
        for (attr_name, extra_msg) in [('upload_dir', '\nPlease specify `train.RunConfig(storage_path)` instead.'), ('syncer', '\nPlease implement custom syncing logic with a custom `pyarrow.fs.FileSystem` instead, and pass it into `train.RunConfig(storage_filesystem)`. See here: https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html#custom-storage'), ('sync_on_checkpoint', '')]:
            self._deprecation_warning(attr_name, extra_msg)

    def _repr_html_(self) -> str:
        if False:
            return 10
        'Generate an HTML representation of the SyncConfig.'
        return Template('scrollableTable.html.j2').render(table=tabulate({'Setting': ['Sync period', 'Sync timeout'], 'Value': [self.sync_period, self.sync_timeout]}, tablefmt='html', showindex=False, headers='keys'), max_height='none')

class _BackgroundProcess:

    def __init__(self, fn: Callable):
        if False:
            print('Hello World!')
        self._fn = fn
        self._process = None
        self._result = {}
        self._start_time = float('-inf')

    @property
    def is_running(self):
        if False:
            while True:
                i = 10
        return self._process and self._process.is_alive()

    @property
    def start_time(self):
        if False:
            print('Hello World!')
        return self._start_time

    def start(self, *args, **kwargs):
        if False:
            return 10
        if self.is_running:
            return False
        self._result = {}

        def entrypoint():
            if False:
                i = 10
                return i + 15
            try:
                result = self._fn(*args, **kwargs)
            except Exception as e:
                self._result['exception'] = e
                return
            self._result['result'] = result
        self._process = threading.Thread(target=entrypoint)
        self._process.daemon = True
        self._process.start()
        self._start_time = time.time()

    def wait(self, timeout: Optional[float]=None) -> Any:
        if False:
            print('Hello World!')
        'Waits for the background process to finish running. Waits until the\n        background process has run for at least `timeout` seconds, counting from\n        the time when the process was started.'
        if not self._process:
            return None
        time_remaining = None
        if timeout:
            elapsed = time.time() - self.start_time
            time_remaining = max(timeout - elapsed, 0)
        self._process.join(timeout=time_remaining)
        if self._process.is_alive():
            self._process = None
            raise TimeoutError(f"{getattr(self._fn, '__name__', str(self._fn))} did not finish running within the timeout of {timeout} seconds.")
        self._process = None
        exception = self._result.get('exception')
        if exception:
            raise exception
        result = self._result.get('result')
        self._result = {}
        return result

class Syncer(abc.ABC):
    """Syncer class for synchronizing data between Ray nodes and remote (cloud) storage.

    This class handles data transfer for two cases:

    1. Synchronizing data such as experiment checkpoints from the driver to
       cloud storage.
    2. Synchronizing data such as trial checkpoints from remote trainables to
       cloud storage.

    Synchronizing tasks are usually asynchronous and can be awaited using ``wait()``.
    The base class implements a ``wait_or_retry()`` API that will retry a failed
    sync command.

    The base class also exposes an API to only kick off syncs every ``sync_period``
    seconds.

    Args:
        sync_period: The minimum time in seconds between sync operations, as
            used by ``sync_up/down_if_needed``.
        sync_timeout: The maximum time to wait for a sync process to finish before
            issuing a new sync operation. Ex: should be used by ``wait`` if launching
            asynchronous sync tasks.
    """

    def __init__(self, sync_period: float=DEFAULT_SYNC_PERIOD, sync_timeout: float=DEFAULT_SYNC_TIMEOUT):
        if False:
            return 10
        self.sync_period = sync_period
        self.sync_timeout = sync_timeout
        self.last_sync_up_time = float('-inf')
        self.last_sync_down_time = float('-inf')

    @abc.abstractmethod
    def sync_up(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Synchronize local directory to remote directory.\n\n        This function can spawn an asynchronous process that can be awaited in\n        ``wait()``.\n\n        Args:\n            local_dir: Local directory to sync from.\n            remote_dir: Remote directory to sync up to. This is an URI\n                (``protocol://remote/path``).\n            exclude: Pattern of files to exclude, e.g.\n                ``["*/checkpoint_*]`` to exclude trial checkpoints.\n\n        Returns:\n            True if sync process has been spawned, False otherwise.\n\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def sync_down(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None) -> bool:
        if False:
            while True:
                i = 10
        'Synchronize remote directory to local directory.\n\n        This function can spawn an asynchronous process that can be awaited in\n        ``wait()``.\n\n        Args:\n            remote_dir: Remote directory to sync down from. This is an URI\n                (``protocol://remote/path``).\n            local_dir: Local directory to sync to.\n            exclude: Pattern of files to exclude, e.g.\n                ``["*/checkpoint_*]`` to exclude trial checkpoints.\n\n        Returns:\n            True if sync process has been spawned, False otherwise.\n\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, remote_dir: str) -> bool:
        if False:
            return 10
        'Delete directory on remote storage.\n\n        This function can spawn an asynchronous process that can be awaited in\n        ``wait()``.\n\n        Args:\n            remote_dir: Remote directory to delete. This is an URI\n                (``protocol://remote/path``).\n\n        Returns:\n            True if sync process has been spawned, False otherwise.\n\n        '
        raise NotImplementedError

    def retry(self):
        if False:
            for i in range(10):
                print('nop')
        'Retry the last sync up, sync down, or delete command.\n\n        You should implement this method if you spawn asynchronous syncing\n        processes.\n        '
        pass

    def wait(self):
        if False:
            return 10
        'Wait for asynchronous sync command to finish.\n\n        You should implement this method if you spawn asynchronous syncing\n        processes. This method should timeout after the asynchronous command\n        has run for `sync_timeout` seconds and raise a `TimeoutError`.\n        '
        pass

    def sync_up_if_needed(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
        if False:
            while True:
                i = 10
        'Syncs up if time since last sync up is greater than sync_period.\n\n        Args:\n            local_dir: Local directory to sync from.\n            remote_dir: Remote directory to sync up to. This is an URI\n                (``protocol://remote/path``).\n            exclude: Pattern of files to exclude, e.g.\n                ``["*/checkpoint_*]`` to exclude trial checkpoints.\n        '
        now = time.time()
        if now - self.last_sync_up_time >= self.sync_period:
            result = self.sync_up(local_dir=local_dir, remote_dir=remote_dir, exclude=exclude)
            self.last_sync_up_time = now
            return result

    def sync_down_if_needed(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None):
        if False:
            print('Hello World!')
        'Syncs down if time since last sync down is greater than sync_period.\n\n        Args:\n            remote_dir: Remote directory to sync down from. This is an URI\n                (``protocol://remote/path``).\n            local_dir: Local directory to sync to.\n            exclude: Pattern of files to exclude, e.g.\n                ``["*/checkpoint_*]`` to exclude trial checkpoints.\n        '
        now = time.time()
        if now - self.last_sync_down_time >= self.sync_period:
            result = self.sync_down(remote_dir=remote_dir, local_dir=local_dir, exclude=exclude)
            self.last_sync_down_time = now
            return result

    def wait_or_retry(self, max_retries: int=2, backoff_s: int=5):
        if False:
            for i in range(10):
                print('nop')
        assert max_retries > 0
        last_error_traceback = None
        for i in range(max_retries + 1):
            try:
                self.wait()
            except Exception as e:
                attempts_remaining = max_retries - i
                if attempts_remaining == 0:
                    last_error_traceback = traceback.format_exc()
                    break
                logger.error(f'The latest sync operation failed with the following error: {repr(e)}\nRetrying {attempts_remaining} more time(s) after sleeping for {backoff_s} seconds...')
                time.sleep(backoff_s)
                self.retry()
                continue
            return
        raise RuntimeError(f'Failed sync even after {max_retries} retries. The latest sync failed with the following error:\n{last_error_traceback}')

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.last_sync_up_time = float('-inf')
        self.last_sync_down_time = float('-inf')

    def close(self):
        if False:
            while True:
                i = 10
        pass

    def _repr_html_(self) -> str:
        if False:
            while True:
                i = 10
        return

class _BackgroundSyncer(Syncer):
    """Syncer using a background process for asynchronous file transfer."""

    def __init__(self, sync_period: float=DEFAULT_SYNC_PERIOD, sync_timeout: float=DEFAULT_SYNC_TIMEOUT):
        if False:
            return 10
        super(_BackgroundSyncer, self).__init__(sync_period=sync_period, sync_timeout=sync_timeout)
        self._sync_process = None
        self._current_cmd = None

    def _should_continue_existing_sync(self):
        if False:
            while True:
                i = 10
        'Returns whether a previous sync is still running within the timeout.'
        return self._sync_process and self._sync_process.is_running and (time.time() - self._sync_process.start_time < self.sync_timeout)

    def _launch_sync_process(self, sync_command: Tuple[Callable, Dict]):
        if False:
            for i in range(10):
                print('nop')
        'Waits for the previous sync process to finish,\n        then launches a new process that runs the given command.'
        if self._sync_process:
            try:
                self.wait()
            except Exception:
                logger.warning(f'Last sync command failed with the following error:\n{traceback.format_exc()}')
        self._current_cmd = sync_command
        self.retry()

    def sync_up(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
        if False:
            return 10
        if self._should_continue_existing_sync():
            logger.warning(f'Last sync still in progress, skipping sync up of {local_dir} to {remote_dir}')
            return False
        sync_up_cmd = self._sync_up_command(local_path=local_dir, uri=remote_dir, exclude=exclude)
        self._launch_sync_process(sync_up_cmd)
        return True

    def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List]=None) -> Tuple[Callable, Dict]:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def sync_down(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None) -> bool:
        if False:
            i = 10
            return i + 15
        if self._should_continue_existing_sync():
            logger.warning(f'Last sync still in progress, skipping sync down of {remote_dir} to {local_dir}')
            return False
        sync_down_cmd = self._sync_down_command(uri=remote_dir, local_path=local_dir)
        self._launch_sync_process(sync_down_cmd)
        return True

    def _sync_down_command(self, uri: str, local_path: str) -> Tuple[Callable, Dict]:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def delete(self, remote_dir: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self._should_continue_existing_sync():
            logger.warning(f'Last sync still in progress, skipping deletion of {remote_dir}')
            return False
        delete_cmd = self._delete_command(uri=remote_dir)
        self._launch_sync_process(delete_cmd)
        return True

    def _delete_command(self, uri: str) -> Tuple[Callable, Dict]:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def wait(self):
        if False:
            print('Hello World!')
        if self._sync_process:
            try:
                self._sync_process.wait(timeout=self.sync_timeout)
            except Exception as e:
                raise e
            finally:
                self._sync_process = None

    def retry(self):
        if False:
            return 10
        if not self._current_cmd:
            raise RuntimeError('No sync command set, cannot retry.')
        (cmd, kwargs) = self._current_cmd
        self._sync_process = _BackgroundProcess(cmd)
        self._sync_process.start(**kwargs)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = self.__dict__.copy()
        state['_sync_process'] = None
        return state