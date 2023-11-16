import concurrent.futures
import pathlib
import threading
from threading import Thread
from time import time
from typing import TYPE_CHECKING, Optional, Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from lightning.app.core.queues import BaseQueue
from lightning.app.storage.path import _filesystem
from lightning.app.storage.requests import _ExistsRequest, _GetRequest
from lightning.app.utilities.app_helpers import Logger
_PathRequest = Union[_GetRequest, _ExistsRequest]
_logger = Logger(__name__)
num_workers = 8
if TYPE_CHECKING:
    import lightning.app

class _Copier(Thread):
    """The Copier is a thread running alongside a LightningWork.

    It maintains two queues that connect to the central
    :class:`~lightning.app.storage.orchestrator.StorageOrchestrator`,
    the request queue and the response queue. The Copier waits for a request to be pushed to the request queue,
    processes it and sends back the request through the response queue. In the current implementation, the Copier
    simply copies the requested file from the local filesystem to a shared directory (determined by
    :func:`~lightning.app.storage.path.shared_storage_path`). Any errors raised during the copy will be added to the
    response and get re-raised within the corresponding LightningWork.

    Args:
        copy_request_queue: A queue connecting the central StorageOrchestrator with the Copier. The orchestrator
            will send requests to this queue.
        copy_response_queue: A queue connecting the central StorageOrchestrator with the Copier. The Copier
            will send a response to this queue whenever a requested copy has finished.

    """

    def __init__(self, work: 'lightning.app.LightningWork', copy_request_queue: 'BaseQueue', copy_response_queue: 'BaseQueue') -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(daemon=True)
        self._work = work
        self.copy_request_queue = copy_request_queue
        self.copy_response_queue = copy_response_queue
        self._exit_event = threading.Event()
        self._sleep_time = 0.1

    def run(self) -> None:
        if False:
            print('Hello World!')
        while not self._exit_event.is_set():
            self._exit_event.wait(self._sleep_time)
            self.run_once()

    def join(self, timeout: Optional[float]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._exit_event.set()
        super().join(timeout)

    def run_once(self):
        if False:
            print('Hello World!')
        request: _PathRequest = self.copy_request_queue.get()
        t0 = time()
        obj: Optional[lightning.app.storage.path.Path] = _find_matching_path(self._work, request)
        if obj is None:
            obj: lightning.app.storage.Payload = getattr(self._work, request.name)
        if isinstance(request, _ExistsRequest):
            response = obj._handle_exists_request(self._work, request)
        elif isinstance(request, _GetRequest):
            response = obj._handle_get_request(self._work, request)
        else:
            raise TypeError(f'The file copy request had an invalid type. Expected PathGetRequest or PathExistsRequest, got: {type(request)}')
        response.timedelta = time() - t0
        self.copy_response_queue.put(response)

def _find_matching_path(work, request: _GetRequest) -> Optional['lightning.app.storage.path.Path']:
    if False:
        for i in range(10):
            print('nop')
    for name in work._paths:
        candidate: lightning.app.storage.path.Path = getattr(work, name)
        if candidate.hash == request.hash:
            return candidate
    return None

def _copy_files(source_path: pathlib.Path, destination_path: pathlib.Path, fs: Optional[AbstractFileSystem]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Copy files from one path to another.\n\n    The source path must either be an existing file or folder. If the source is a folder, the destination path is\n    interpreted as a folder as well. If the source is a file, the destination path is interpreted as a file too.\n\n    Files in a folder are copied recursively and efficiently using multiple threads.\n\n    '
    if fs is None:
        fs = _filesystem()

    def _copy(from_path: pathlib.Path, to_path: pathlib.Path) -> Optional[Exception]:
        if False:
            return 10
        _logger.debug(f'Copying {str(from_path)} -> {str(to_path)}')
        try:
            if isinstance(fs, LocalFileSystem):
                fs.makedirs(str(to_path.parent), exist_ok=True)
            fs.put(str(from_path), str(to_path), recursive=False)
        except Exception as ex:
            return ex
    if source_path.is_dir():
        src = [file for file in source_path.rglob('*') if file.is_file()]
        dst = [destination_path / file.relative_to(source_path) for file in src]
        with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            results = executor.map(_copy, src, dst)
        exception = next((e for e in results if isinstance(e, Exception)), None)
        if exception:
            raise exception
    else:
        if isinstance(fs, LocalFileSystem):
            fs.makedirs(str(destination_path.parent), exist_ok=True)
        fs.put(str(source_path), str(destination_path))