import os
import selectors
import sys
from threading import Event
from typing import Iterator

class InputReader:
    """Read input from stdin."""

    def __init__(self, timeout: float=0.1) -> None:
        if False:
            return 10
        '\n\n        Args:\n            timeout: Seconds to block for input.\n        '
        self._fileno = sys.__stdin__.fileno()
        self.timeout = timeout
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._fileno, selectors.EVENT_READ)
        self._exit_event = Event()

    def more_data(self) -> bool:
        if False:
            while True:
                i = 10
        'Check if there is data pending.'
        EVENT_READ = selectors.EVENT_READ
        for (_key, events) in self._selector.select(0.01):
            if events & EVENT_READ:
                return True
        return False

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Close the reader (will exit the iterator).'
        self._exit_event.set()

    def __iter__(self) -> Iterator[bytes]:
        if False:
            return 10
        'Read input, yield bytes.'
        fileno = self._fileno
        read = os.read
        exit_set = self._exit_event.is_set
        EVENT_READ = selectors.EVENT_READ
        while not exit_set():
            for (_key, events) in self._selector.select(self.timeout):
                if events & EVENT_READ:
                    data = read(fileno, 1024)
                    if not data:
                        return
                    yield data