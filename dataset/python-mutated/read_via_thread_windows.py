"""On Windows we use overlapped mechanism, borrowing it from asyncio (but without the event loop)."""
from __future__ import annotations
import logging
from asyncio.windows_utils import BUFSIZE
from time import sleep
from typing import Callable
import _overlapped
from .read_via_thread import ReadViaThread

class ReadViaThreadWindows(ReadViaThread):

    def __init__(self, file_no: int, handler: Callable[[bytes], None], name: str, drain: bool) -> None:
        if False:
            print('Hello World!')
        super().__init__(file_no, handler, name, drain)
        self.closed = False
        self._ov: _overlapped.Overlapped | None = None
        self._waiting_for_read = False

    def _read_stream(self) -> None:
        if False:
            return 10
        keep_reading = True
        while keep_reading:
            wait = self._read_batch()
            if wait is None:
                break
            if wait is True:
                sleep(0.01)
            keep_reading = not self.stop.is_set()

    def _drain_stream(self) -> None:
        if False:
            print('Hello World!')
        wait: bool | None = self.closed
        while wait is False:
            wait = self._read_batch()

    def _read_batch(self) -> bool | None:
        if False:
            for i in range(10):
                print('nop')
        ':returns: None means error can no longer read, True wait for result, False try again'
        if self._waiting_for_read is False:
            self._ov = _overlapped.Overlapped(0)
            try:
                self._ov.ReadFile(self.file_no, BUFSIZE)
                self._waiting_for_read = True
            except OSError:
                self.closed = True
                return None
        try:
            data = self._ov.getresult(False)
        except OSError as exception:
            win_error = getattr(exception, 'winerror', None)
            if win_error == 996:
                return True
            if win_error != 995:
                logging.error('failed to read %r', exception)
            return None
        else:
            self._ov = None
            self._waiting_for_read = False
            if data:
                self.handler(data)
            else:
                return None
        return False