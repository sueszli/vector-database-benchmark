import logging
from enum import Enum
from typing import Any, Callable, Optional, Set
from pydantic import BaseModel
from e2b.constants import TIMEOUT
from e2b.sandbox.exception import FilesystemException, RpcException
from e2b.sandbox.sandbox_connection import SandboxConnection
from e2b.utils.str import snake_case_to_camel_case
logger = logging.getLogger(__name__)

class FilesystemOperation(str, Enum):
    Create = 'Create'
    Write = 'Write'
    Remove = 'Remove'
    Rename = 'Rename'
    Chmod = 'Chmod'

class FilesystemEvent(BaseModel):
    path: str
    name: str
    operation: FilesystemOperation
    timestamp: int
    '\n    Unix epoch in nanoseconds\n    '
    is_dir: bool

    class Config:
        alias_generator = snake_case_to_camel_case

class FilesystemWatcher:

    @property
    def path(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        The path to watch.\n        '
        return self._path

    def __init__(self, connection: SandboxConnection, path: str, service_name: str):
        if False:
            return 10
        self._connection = connection
        self._path = path
        self._service_name = service_name
        self._unsubscribe: Optional[Callable[[], Any]] = None
        self._listeners: Set[Callable[[FilesystemEvent], Any]] = set()

    def start(self, timeout: Optional[float]=TIMEOUT) -> None:
        if False:
            print('Hello World!')
        '\n        Start the filesystem watcher.\n\n        :param timeout: Specify the duration, in seconds to give the method to finish its execution before it times out (default is 60 seconds). If set to None, the method will continue to wait until it completes, regardless of time\n        '
        if self._unsubscribe:
            return
        logger.debug('Starting filesystem watcher for %s', self.path)
        try:
            self._unsubscribe = self._connection._subscribe(self._service_name, self._handle_filesystem_events, 'watchDir', self.path, timeout=timeout)
            logger.debug('Started filesystem watcher for %s', self.path)
        except RpcException as e:
            raise FilesystemException(e.message) from e

    def stop(self) -> None:
        if False:
            print('Hello World!')
        '\n        Stop the filesystem watcher.\n        '
        logger.debug('Stopping filesystem watcher for %s', self.path)
        self._listeners.clear()
        if self._unsubscribe:
            try:
                self._unsubscribe()
                self._unsubscribe = None
                logger.debug('Stopped filesystem watcher for %s', self.path)
            except RpcException as e:
                raise FilesystemException(e.message) from e

    def add_event_listener(self, listener: Callable[[FilesystemEvent], Any]):
        if False:
            return 10
        '\n        Add a listener for filesystem events.\n\n        :param listener: Listener to add\n\n        :return: Function that removes the listener\n        '
        logger.debug('Adding filesystem watcher listener for %s', self.path)
        self._listeners.add(listener)

        def delete_listener() -> None:
            if False:
                for i in range(10):
                    print('nop')
            self._listeners.remove(listener)
        return delete_listener

    def _handle_filesystem_events(self, event: dict) -> None:
        if False:
            while True:
                i = 10
        event = FilesystemEvent(**event)
        for listener in self._listeners:
            listener(event)