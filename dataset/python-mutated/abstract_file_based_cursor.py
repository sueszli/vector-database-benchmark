import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, MutableMapping
from airbyte_cdk.sources.file_based.config.file_based_stream_config import FileBasedStreamConfig
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from airbyte_cdk.sources.file_based.types import StreamState

class AbstractFileBasedCursor(ABC):
    """
    Abstract base class for cursors used by file-based streams.
    """

    @abstractmethod
    def __init__(self, stream_config: FileBasedStreamConfig, **kwargs: Any):
        if False:
            while True:
                i = 10
        '\n        Common interface for all cursors.\n        '
        ...

    @abstractmethod
    def add_file(self, file: RemoteFile) -> None:
        if False:
            return 10
        '\n        Add a file to the cursor. This method is called when a file is processed by the stream.\n        :param file: The file to add\n        '
        ...

    @abstractmethod
    def set_initial_state(self, value: StreamState) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Set the initial state of the cursor. The cursor cannot be initialized at construction time because the stream doesn't know its state yet.\n        :param value: The stream state\n        "

    @abstractmethod
    def get_state(self) -> MutableMapping[str, Any]:
        if False:
            return 10
        '\n        Get the state of the cursor.\n        '
        ...

    @abstractmethod
    def get_start_time(self) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the start time of the current sync.\n        '
        ...

    @abstractmethod
    def get_files_to_sync(self, all_files: Iterable[RemoteFile], logger: logging.Logger) -> Iterable[RemoteFile]:
        if False:
            return 10
        '\n        Given the list of files in the source, return the files that should be synced.\n        :param all_files: All files in the source\n        :param logger:\n        :return: The files that should be synced\n        '
        ...