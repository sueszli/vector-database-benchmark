"""
Provides ByteQueue, a high-performance queue for bytes objects.
"""
from collections import deque
from bisect import bisect
from typing import Generator

class ByteQueue:
    """
    Queue for bytes
    Can append bytes objects at the right,
    and pop arbitrary-size bytes objects at the left.

    The naive implementation would look like this:

    append(self, data):
        self.buf += data

    popleft(self, size):
        result, self.buf = self.buf[:size], self.buf[size:]
        return result

    However, due to python's nature, that would be extremely slow.

    Thus, the bytes objects are actually stored - unmodified -
    in an internal queue, and only concatenated during popleft().
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.bufs = deque()
        self.size = 0

    def __len__(self):
        if False:
            return 10
        '\n        Size of all currently-stored data.\n        '
        return self.size

    def append(self, data: bytes) -> None:
        if False:
            print('Hello World!')
        '\n        Adds bytes to the buffer.\n        '
        if not isinstance(data, bytes):
            raise TypeError('expected a bytes object, but got ' + repr(data))
        self.bufs.append(data)
        self.size += len(data)

    def popleft(self, size: int) -> bytes:
        if False:
            print('Hello World!')
        '\n        Returns the requested amount of bytes from the buffer.\n        '
        if size > self.size:
            raise ValueError('ByteQueue does not contain enough bytes')
        self.size -= size
        resultbufs = []
        required = size
        while required > 0:
            buf = self.bufs.popleft()
            resultbufs.append(buf)
            required -= len(buf)
        if required < 0:
            buf = resultbufs.pop()
            popped = buf[:required]
            kept = buf[required:]
            resultbufs.append(popped)
            self.bufs.appendleft(kept)
        return b''.join(resultbufs)

class ByteBuffer:
    """
    Similar to ByteQueue, but instead of popleft, allows reading random slices,
    and trimleft, which discards data from the left.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.bufs = [None]
        self.index = [0]
        self.discardedbufs = 1
        self.discardedbytes = 0

    def __len__(self):
        if False:
            print('Hello World!')
        return self.index[-1]

    def append(self, data: bytes) -> None:
        if False:
            i = 10
            return i + 15
        '\n        appends new data to the right of the buffer\n        '
        if not isinstance(data, bytes):
            raise TypeError('expected bytes, but got ' + repr(data))
        self.bufs.append(data)
        self.index.append(len(self) + len(data))

    def discardleft(self, keep: int) -> None:
        if False:
            print('Hello World!')
        "\n        discards data at the beginning of the buffer.\n        keeps at least the 'keep' most recent bytes.\n        "
        discardamount = len(self) - keep
        if discardamount <= self.discardedbytes:
            return
        discardto = bisect(self.index, discardamount)
        for idx in range(self.discardedbufs, discardto):
            self.bufs[idx] = None
        self.discardedbufs = discardto
        self.discardedbytes = discardamount

    def hasbeendiscarded(self, position: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        returns True if the given position has already been discarded\n        and is no longer valid.\n        '
        return position < self.discardedbytes

    def __getitem__(self, pos):
        if False:
            while True:
                i = 10
        '\n        a slice with default stepping is required.\n\n        when attempting to access already-discarded data, a ValueError\n        is raised.\n        '
        if not isinstance(pos, slice):
            raise TypeError('expected slice')
        if pos.step is not None:
            raise TypeError('slicing with steps is not supported')
        (start, end) = (pos.start, pos.stop)
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        if start < 0:
            start = len(self) + start
        if end < 0:
            end = len(self) + end
        return b''.join(self.get_buffers(start, end))

    class DiscardedError(Exception):
        """
        raised by get_buffers and the indexing operator if the requested
        data has already been discarded.
        """

    def get_buffers(self, start: int, end: int) -> Generator[bytes, None, None]:
        if False:
            return 10
        '\n        yields any amount of bytes objects that constitute the data\n        between start and end.\n\n        used internally by __getitem__, but may be useful externally\n        as well.\n\n        performs bounds checking, but end must be greater than start.\n        '
        start = max(start, 0)
        end = min(end, len(self))
        if end <= start:
            yield b''
            return
        if self.hasbeendiscarded(start):
            raise self.DiscardedError(start, end)
        idx = bisect(self.index, start)
        buf = self.bufs[idx]
        buf = buf[start - self.index[idx]:]
        remaining = end - start
        while remaining > len(buf):
            remaining -= len(buf)
            yield buf
            idx += 1
            buf = self.bufs[idx]
        buf = buf[:remaining]
        yield buf