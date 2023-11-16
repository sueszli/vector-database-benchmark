import logging
from typing import Union, Optional, TYPE_CHECKING
from aim.sdk.base_run import BaseRun
from aim.ext.cleanup import AutoClean
if TYPE_CHECKING:
    from aim.sdk import Repo
logger = logging.getLogger(__name__)

class RunAutoClean(AutoClean['MaintenanceRun']):
    PRIORITY = 90

    def __init__(self, instance: 'MaintenanceRun') -> None:
        if False:
            while True:
                i = 10
        '\n        Prepare the `Run` for automatic cleanup.\n\n        Args:\n            instance: The `Run` instance to be cleaned up.\n        '
        super().__init__(instance)
        self._lock = instance._lock

    def _close(self) -> None:
        if False:
            print('Hello World!')
        if self._lock is not None:
            self._lock.release()

class MaintenanceRun(BaseRun):

    def __init__(self, run_hash: str, repo: Optional[Union[str, 'Repo']]=None):
        if False:
            print('Hello World!')
        self._resources: Optional[RunAutoClean] = None
        super().__init__(run_hash, repo=repo, read_only=False)
        self._resources = RunAutoClean(self)