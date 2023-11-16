"""A class that watches a given path via polling."""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
LOGGER = get_logger(__name__)
_MAX_WORKERS = 4
_POLLING_PERIOD_SECS = 0.2

class PollingPathWatcher:
    """Watches a path on disk via a polling loop."""
    _executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

    @staticmethod
    def close_all() -> None:
        if False:
            for i in range(10):
                print('nop')
        'Close top-level watcher object.\n\n        This is a no-op, and exists for interface parity with\n        EventBasedPathWatcher.\n        '
        LOGGER.debug('Watcher closed')

    def __init__(self, path: str, on_changed: Callable[[str], None], *, glob_pattern: Optional[str]=None, allow_nonexistent: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Constructor.\n\n        You do not need to retain a reference to a PollingPathWatcher to\n        prevent it from being garbage collected. (The global _executor object\n        retains references to all active instances.)\n        '
        self._path = path
        self._on_changed = on_changed
        self._glob_pattern = glob_pattern
        self._allow_nonexistent = allow_nonexistent
        self._active = True
        self._modification_time = util.path_modification_time(self._path, self._allow_nonexistent)
        self._md5 = util.calc_md5_with_blocking_retries(self._path, glob_pattern=self._glob_pattern, allow_nonexistent=self._allow_nonexistent)
        self._schedule()

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return repr_(self)

    def _schedule(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def task():
            if False:
                while True:
                    i = 10
            time.sleep(_POLLING_PERIOD_SECS)
            self._check_if_path_changed()
        PollingPathWatcher._executor.submit(task)

    def _check_if_path_changed(self) -> None:
        if False:
            return 10
        if not self._active:
            return
        modification_time = util.path_modification_time(self._path, self._allow_nonexistent)
        if modification_time <= self._modification_time:
            self._schedule()
            return
        self._modification_time = modification_time
        md5 = util.calc_md5_with_blocking_retries(self._path, glob_pattern=self._glob_pattern, allow_nonexistent=self._allow_nonexistent)
        if md5 == self._md5:
            self._schedule()
            return
        self._md5 = md5
        LOGGER.debug('Change detected: %s', self._path)
        self._on_changed(self._path)
        self._schedule()

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Stop watching the file system.'
        self._active = False