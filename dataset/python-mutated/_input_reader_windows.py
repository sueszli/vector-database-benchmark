import os
import sys
from threading import Event
from typing import Iterator

class InputReader:
    """Read input from stdin."""

    def __init__(self, timeout: float=0.1) -> None:
        if False:
            while True:
                i = 10
        '\n\n        Args:\n            timeout: Seconds to block for input.\n        '
        self._fileno = sys.__stdin__.fileno()
        self.timeout = timeout
        self._exit_event = Event()

    def more_data(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if there is data pending.'
        return True

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Close the reader (will exit the iterator).'
        self._exit_event.set()

    def __iter__(self) -> Iterator[bytes]:
        if False:
            return 10
        'Read input, yield bytes.'
        while not self._exit_event.is_set():
            try:
                data = os.read(self._fileno, 1024) or None
            except Exception:
                break
            if not data:
                break
            yield data