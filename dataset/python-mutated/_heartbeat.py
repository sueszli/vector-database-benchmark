import abc
import copy
from threading import Event
from threading import Thread
from types import TracebackType
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
import optuna
from optuna._experimental import experimental_func
from optuna.storages import BaseStorage
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

class BaseHeartbeat(metaclass=abc.ABCMeta):
    """Base class for heartbeat.

    This class is not supposed to be directly accessed by library users.

    The heartbeat mechanism periodically checks whether each trial process is alive during an
    optimization loop. To support this mechanism, the methods of
    :class:`~optuna.storages._heartbeat.BaseHeartbeat` is implemented for the target database
    backend, typically with multiple inheritance of :class:`~optuna.storages._base.BaseStorage`
    and :class:`~optuna.storages._heartbeat.BaseHeartbeat`.

    .. seealso::
        See :class:`~optuna.storages.RDBStorage`, where the backend supports heartbeat.
    """

    @abc.abstractmethod
    def record_heartbeat(self, trial_id: int) -> None:
        if False:
            while True:
                i = 10
        'Record the heartbeat of the trial.\n\n        Args:\n            trial_id:\n                ID of the trial.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_stale_trial_ids(self, study_id: int) -> List[int]:
        if False:
            print('Hello World!')
        'Get the stale trial ids of the study.\n\n        Args:\n            study_id:\n                ID of the study.\n        Returns:\n            List of IDs of trials whose heartbeat has not been updated for a long time.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_heartbeat_interval(self) -> Optional[int]:
        if False:
            return 10
        'Get the heartbeat interval if it is set.\n\n        Returns:\n            The heartbeat interval if it is set, otherwise :obj:`None`.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_failed_trial_callback(self) -> Optional[Callable[['optuna.Study', FrozenTrial], None]]:
        if False:
            for i in range(10):
                print('nop')
        'Get the failed trial callback function.\n\n        Returns:\n            The failed trial callback function if it is set, otherwise :obj:`None`.\n        '
        raise NotImplementedError()

class BaseHeartbeatThread(metaclass=abc.ABCMeta):

    def __enter__(self) -> None:
        if False:
            print('Hello World!')
        self.start()

    def __exit__(self, exc_type: Optional[Type[Exception]], exc_value: Optional[Exception], traceback: Optional[TracebackType]) -> None:
        if False:
            print('Hello World!')
        self.join()

    @abc.abstractmethod
    def start(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def join(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class NullHeartbeatThread(BaseHeartbeatThread):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def start(self) -> None:
        if False:
            print('Hello World!')
        pass

    def join(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

class HeartbeatThread(BaseHeartbeatThread):

    def __init__(self, trial_id: int, heartbeat: BaseHeartbeat) -> None:
        if False:
            while True:
                i = 10
        self._trial_id = trial_id
        self._heartbeat = heartbeat
        self._thread: Optional[Thread] = None
        self._stop_event: Optional[Event] = None

    def start(self) -> None:
        if False:
            while True:
                i = 10
        self._stop_event = Event()
        self._thread = Thread(target=self._record_heartbeat, args=(self._trial_id, self._heartbeat, self._stop_event))
        self._thread.start()

    def join(self) -> None:
        if False:
            print('Hello World!')
        assert self._stop_event is not None
        assert self._thread is not None
        self._stop_event.set()
        self._thread.join()

    @staticmethod
    def _record_heartbeat(trial_id: int, heartbeat: BaseHeartbeat, stop_event: Event) -> None:
        if False:
            print('Hello World!')
        heartbeat_interval = heartbeat.get_heartbeat_interval()
        assert heartbeat_interval is not None
        while True:
            heartbeat.record_heartbeat(trial_id)
            if stop_event.wait(timeout=heartbeat_interval):
                return

def get_heartbeat_thread(trial_id: int, storage: BaseStorage) -> BaseHeartbeatThread:
    if False:
        return 10
    if is_heartbeat_enabled(storage):
        assert isinstance(storage, BaseHeartbeat)
        return HeartbeatThread(trial_id, storage)
    else:
        return NullHeartbeatThread()

@experimental_func('2.9.0')
def fail_stale_trials(study: 'optuna.Study') -> None:
    if False:
        i = 10
        return i + 15
    'Fail stale trials and run their failure callbacks.\n\n    The running trials whose heartbeat has not been updated for a long time will be failed,\n    that is, those states will be changed to :obj:`~optuna.trial.TrialState.FAIL`.\n\n    .. seealso::\n\n        See :class:`~optuna.storages.RDBStorage`.\n\n    Args:\n        study:\n            Study holding the trials to check.\n    '
    storage = study._storage
    if not isinstance(storage, BaseHeartbeat):
        return
    if not is_heartbeat_enabled(storage):
        return
    failed_trial_ids = []
    for trial_id in storage._get_stale_trial_ids(study._study_id):
        try:
            if storage.set_trial_state_values(trial_id, state=TrialState.FAIL):
                failed_trial_ids.append(trial_id)
        except RuntimeError:
            pass
    failed_trial_callback = storage.get_failed_trial_callback()
    if failed_trial_callback is not None:
        for trial_id in failed_trial_ids:
            failed_trial = copy.deepcopy(storage.get_trial(trial_id))
            failed_trial_callback(study, failed_trial)

def is_heartbeat_enabled(storage: BaseStorage) -> bool:
    if False:
        print('Hello World!')
    'Check whether the storage enables the heartbeat.\n\n    Returns:\n        :obj:`True` if the storage also inherits :class:`~optuna.storages._heartbeat.BaseHeartbeat`\n        and the return value of :meth:`~optuna.storages.BaseStorage.get_heartbeat_interval` is an\n        integer, otherwise :obj:`False`.\n    '
    return isinstance(storage, BaseHeartbeat) and storage.get_heartbeat_interval() is not None