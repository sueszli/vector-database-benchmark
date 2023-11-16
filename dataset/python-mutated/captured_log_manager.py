from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import IO, Callable, Generator, Iterator, NamedTuple, Optional, Sequence
from typing_extensions import Final, Self
import dagster._check as check
from dagster._core.storage.compute_log_manager import ComputeIOType
MAX_BYTES_CHUNK_READ: Final = 4194304

class CapturedLogContext(NamedTuple('_CapturedLogContext', [('log_key', Sequence[str]), ('external_url', Optional[str]), ('external_stdout_url', Optional[str]), ('external_stderr_url', Optional[str])])):
    """Object representing the context in which logs are captured.  Can be used by external logging
    sidecar implementations to point the Dagster UI to an external url to view compute logs instead of a
    Dagster-managed location.
    """

    def __new__(cls, log_key: Sequence[str], external_stdout_url: Optional[str]=None, external_stderr_url: Optional[str]=None, external_url: Optional[str]=None):
        if False:
            while True:
                i = 10
        if external_url and (external_stdout_url or external_stderr_url):
            check.failed('Cannot specify both `external_url` and one of `external_stdout_url`/`external_stderr_url`')
        return super(CapturedLogContext, cls).__new__(cls, log_key, external_stdout_url=external_stdout_url, external_stderr_url=external_stderr_url, external_url=external_url)

class CapturedLogData(NamedTuple('_CapturedLogData', [('log_key', Sequence[str]), ('stdout', Optional[bytes]), ('stderr', Optional[bytes]), ('cursor', Optional[str])])):
    """Object representing captured log data, either a partial chunk of the log data or the full
    capture.  Contains the raw bytes and optionally the cursor offset for the partial chunk.
    """

    def __new__(cls, log_key: Sequence[str], stdout: Optional[bytes]=None, stderr: Optional[bytes]=None, cursor: Optional[str]=None):
        if False:
            while True:
                i = 10
        return super(CapturedLogData, cls).__new__(cls, log_key, stdout, stderr, cursor)

class CapturedLogMetadata(NamedTuple('_CapturedLogMetadata', [('stdout_location', Optional[str]), ('stderr_location', Optional[str]), ('stdout_download_url', Optional[str]), ('stderr_download_url', Optional[str])])):
    """Object representing metadata info for the captured log data, containing a display string for
    the location of the log data and a URL for direct download of the captured log data.
    """

    def __new__(cls, stdout_location: Optional[str]=None, stderr_location: Optional[str]=None, stdout_download_url: Optional[str]=None, stderr_download_url: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        return super(CapturedLogMetadata, cls).__new__(cls, stdout_location=stdout_location, stderr_location=stderr_location, stdout_download_url=stdout_download_url, stderr_download_url=stderr_download_url)

class CapturedLogSubscription:

    def __init__(self, manager: 'CapturedLogManager', log_key: Sequence[str], cursor: Optional[str]):
        if False:
            return 10
        self._manager = manager
        self._log_key = log_key
        self._cursor = cursor
        self._observer: Optional[Callable[[CapturedLogData], None]] = None
        self.is_complete = False

    def __call__(self, observer: Optional[Callable[[CapturedLogData], None]]) -> Self:
        if False:
            print('Hello World!')
        self._observer = observer
        self.fetch()
        if self._manager.is_capture_complete(self._log_key):
            self.complete()
        return self

    @property
    def log_key(self) -> Sequence[str]:
        if False:
            while True:
                i = 10
        return self._log_key

    def dispose(self) -> None:
        if False:
            while True:
                i = 10
        self._observer = None
        self._manager.unsubscribe(self)

    def fetch(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self._observer:
            return
        should_fetch = True
        while should_fetch:
            log_data = self._manager.get_log_data(self._log_key, self._cursor, max_bytes=MAX_BYTES_CHUNK_READ)
            if not self._cursor or log_data.cursor != self._cursor:
                self._observer(log_data)
                self._cursor = log_data.cursor
            should_fetch = _has_max_data(log_data.stdout) or _has_max_data(log_data.stderr)

    def complete(self) -> None:
        if False:
            while True:
                i = 10
        self.is_complete = True

def _has_max_data(chunk: Optional[bytes]) -> bool:
    if False:
        return 10
    return chunk and len(chunk) >= MAX_BYTES_CHUNK_READ

class CapturedLogManager(ABC):
    """Abstract base class for capturing the unstructured logs (stdout/stderr) in the current
    process, stored / retrieved with a provided log_key.
    """

    @abstractmethod
    @contextmanager
    def capture_logs(self, log_key: Sequence[str]) -> Generator[CapturedLogContext, None, None]:
        if False:
            return 10
        'Context manager for capturing the stdout/stderr within the current process, and persisting\n        it under the given log key.\n\n        Args:\n            log_key (List[String]): The log key identifying the captured logs\n        '

    @abstractmethod
    @contextmanager
    def open_log_stream(self, log_key: Sequence[str], io_type: ComputeIOType) -> Iterator[Optional[IO[bytes]]]:
        if False:
            print('Hello World!')
        'Context manager for providing an IO stream that enables the caller to write to a log stream\n        managed by the captured log manager, to be read later using the given log key.\n\n        Args:\n            log_key (List[String]): The log key identifying the captured logs\n        '

    @abstractmethod
    def is_capture_complete(self, log_key: Sequence[str]) -> bool:
        if False:
            while True:
                i = 10
        'Flag indicating when the log capture for a given log key has completed.\n\n        Args:\n            log_key (List[String]): The log key identifying the captured logs\n\n        Returns:\n            Boolean\n        '

    @abstractmethod
    def get_log_data(self, log_key: Sequence[str], cursor: Optional[str]=None, max_bytes: Optional[int]=None) -> CapturedLogData:
        if False:
            print('Hello World!')
        'Returns a chunk of the captured stdout logs for a given log key.\n\n        Args:\n            log_key (List[String]): The log key identifying the captured logs\n            cursor (Optional[str]): A cursor representing the position of the log chunk to fetch\n            max_bytes (Optional[int]): A limit on the size of the log chunk to fetch\n\n        Returns:\n            CapturedLogData\n        '

    @abstractmethod
    def get_log_metadata(self, log_key: Sequence[str]) -> CapturedLogMetadata:
        if False:
            print('Hello World!')
        'Returns the metadata of the captured logs for a given log key, including\n        displayable information on where the logs are persisted.\n\n        Args:\n            log_key (List[String]): The log key identifying the captured logs\n\n        Returns:\n            CapturedLogMetadata\n        '

    @abstractmethod
    def delete_logs(self, log_key: Optional[Sequence[str]]=None, prefix: Optional[Sequence[str]]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Deletes the captured logs for a given log key.\n\n        Args:\n            log_key(Optional[List[String]]): The log key of the logs to delete\n            prefix(Optional[List[String]]): The prefix of the log keys to delete\n        '

    @abstractmethod
    def subscribe(self, log_key: Sequence[str], cursor: Optional[str]=None) -> CapturedLogSubscription:
        if False:
            for i in range(10):
                print('nop')
        'Registers an observable object for log data.\n\n        Args:\n            log_key (List[String]): The log key identifying the captured logs\n            cursor (Optional[String]): The string cursor marking the position within the log stream\n        Returns:\n            ComputeLogSubscription\n        '

    @abstractmethod
    def unsubscribe(self, subscription: CapturedLogSubscription) -> None:
        if False:
            print('Hello World!')
        'Deregisters an observable object from receiving log updates.\n\n        Args:\n            subscription (CapturedLogSubscription): subscription object which manages when to send\n                back data to the subscriber\n        '

    def build_log_key_for_run(self, run_id: str, step_key: str) -> Sequence[str]:
        if False:
            return 10
        'Legacy adapter to translate run_id/key to captured log manager-based log_key.'
        return [run_id, 'compute_logs', step_key]