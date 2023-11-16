from __future__ import annotations
import threading
from queue import Queue
from typing import IO
from typing_extensions import Final
MAX_QUEUED_WRITES: Final[int] = 30

class WriterThread(threading.Thread):
    """A thread / file-like to do writes to stdout in the background."""

    def __init__(self, file: IO[str]) -> None:
        if False:
            while True:
                i = 10
        super().__init__(daemon=True)
        self._queue: Queue[str | None] = Queue(MAX_QUEUED_WRITES)
        self._file = file

    def write(self, text: str) -> None:
        if False:
            print('Hello World!')
        'Write text. Text will be enqueued for writing.\n\n        Args:\n            text: Text to write to the file.\n        '
        self._queue.put(text)

    def isatty(self) -> bool:
        if False:
            while True:
                i = 10
        'Pretend to be a terminal.\n\n        Returns:\n            True.\n        '
        return True

    def fileno(self) -> int:
        if False:
            print('Hello World!')
        'Get file handle number.\n\n        Returns:\n            File number of proxied file.\n        '
        return self._file.fileno()

    def flush(self) -> None:
        if False:
            i = 10
            return i + 15
        'Flush the file (a no-op, because flush is done in the thread).'
        return

    def run(self) -> None:
        if False:
            print('Hello World!')
        'Run the thread.'
        write = self._file.write
        flush = self._file.flush
        get = self._queue.get
        qsize = self._queue.qsize
        while True:
            text: str | None = get()
            if text is None:
                break
            write(text)
            if qsize() == 0:
                flush()
        flush()

    def stop(self) -> None:
        if False:
            while True:
                i = 10
        'Stop the thread, and block until it finished.'
        self._queue.put(None)
        self.join()