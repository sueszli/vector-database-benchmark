"""
Provides a FileLikeObject that acts like a FIFO.
"""
import os
from io import UnsupportedOperation
from typing import NoReturn
from .abstract import FileLikeObject
from ..bytequeue import ByteQueue

class FIFO(FileLikeObject):
    """
    File-like wrapper around ByteQueue.

    Data written via write() can later be retrieved via read().

    EOF handling is a bit tricky:

    EOF in the data source (the writer) can not be auto-detected, and must
    be manually indicated by calling seteof().

    Only then will read() show the desired behavior for EOF.

    If EOF has not yet been set, read() will raise ValueError if more data
    than currently available has been requested.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.eof = False
        self.queue = ByteQueue()
        self.pos = 0

    def tell(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Warning: Returns the position for reading.\n\n        Due to the FIFO nature, the position for writing is further-advanced.\n        '
        return self.pos

    def tellw(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the position for writing.\n        '
        return self.pos + len(self.queue)

    def seek(self, offset, whence=os.SEEK_SET) -> NoReturn:
        if False:
            i = 10
            return i + 15
        '\n        Unsupported because this is a FIFO.\n        '
        del offset, whence
        raise UnsupportedOperation('unseekable stream')

    def seekable(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the amount of currently-enqueued data.\n        '
        return len(self.queue)

    def seteof(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Declares that no more data will be added using write().\n\n        Note that this does _not_ mean that no more data is available\n        through write; the queue may still hold some data.\n        '
        self.eof = True

    def write(self, data: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Works until seteof() has been called; accepts bytes objects.\n        '
        if self.eof:
            raise ValueError("EOF has been set; can't write more data")
        self.queue.append(data)

    def writable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def read(self, size: int=-1) -> bytes:
        if False:
            i = 10
            return i + 15
        "\n        If seteof() has not been called yet, requesting more data than\n        len(self) raises a ValueError.\n\n        When called without arguments, all currently-enqueued data is\n        returned; if seteof() has not been set yet, this doesn't\n        indicate EOF, though.\n        "
        if size < 0:
            size = len(self.queue)
        elif self.eof and size > len(self.queue):
            size = len(self.queue)
        self.pos += size
        return self.queue.popleft(size)

    def readable(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def get_size(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.queue)

    def flush(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        self.closed = True
        self.queue = None