"""A reader that drain a stream via its file no on a background thread."""
from __future__ import annotations
from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    import sys
    from types import TracebackType
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
WAIT_GENERAL = 0.05

class ReadViaThread(ABC):

    def __init__(self, file_no: int, handler: Callable[[bytes], None], name: str, drain: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.file_no = file_no
        self.stop = Event()
        self.thread = Thread(target=self._read_stream, name=f'tox-r-{name}-{file_no}')
        self.handler = handler
        self._on_exit_drain = drain

    def __enter__(self) -> Self:
        if False:
            i = 10
            return i + 15
        self.thread.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.stop.set()
        while self.thread.is_alive():
            self.thread.join(WAIT_GENERAL)
        self._drain_stream()

    @abstractmethod
    def _read_stream(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def _drain_stream(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError