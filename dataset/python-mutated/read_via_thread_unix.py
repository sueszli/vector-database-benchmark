"""On UNIX we use select.select to ensure we drain in a non-blocking fashion."""
from __future__ import annotations
import errno
import os
import select
from typing import Callable
from .read_via_thread import ReadViaThread
STOP_EVENT_CHECK_PERIODICITY_IN_MS = 0.01

class ReadViaThreadUnix(ReadViaThread):

    def __init__(self, file_no: int, handler: Callable[[bytes], None], name: str, drain: bool) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(file_no, handler, name, drain)

    def _read_stream(self) -> None:
        if False:
            print('Hello World!')
        while not self.stop.is_set():
            if self._read_available() is None:
                break

    def _drain_stream(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        while True:
            if self._read_available(timeout=0) is not True:
                break

    def _read_available(self, timeout: float=STOP_EVENT_CHECK_PERIODICITY_IN_MS) -> bool | None:
        if False:
            print('Hello World!')
        try:
            (ready, __, ___) = select.select([self.file_no], [], [], timeout)
            if ready:
                data = os.read(self.file_no, 1024)
                if data:
                    self.handler(data)
                    return True
        except OSError as exception:
            if exception.errno in (errno.EBADF, errno.EIO):
                return None
            raise
        else:
            return False