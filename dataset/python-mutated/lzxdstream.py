"""
Wraps the LZXDecompressor in a file-like, read-only stream object.
"""
import os
from io import UnsupportedOperation
from typing import NoReturn
from ..util.filelike.readonly import ReadOnlyFileLikeObject
from ..util.bytequeue import ByteQueue
from ..util.math import INF
from .lzxd import LZXDecompressor

class LZXDStream(ReadOnlyFileLikeObject):
    """
    Read-only stream object that wraps LZXDecompressor.

    Constructor arguments:

    @param compressed_file
        The compressed file-like object; must implement only read().
        If seek(0) works on it, reset() works on this object.

    @param window_bits
        Provided as metadata in MSCAB files; see LZXDecompressor.

        Defaults to 21.

    @param reset_interval
        Zero for MSCAB files; see LZXDecompressor.

        Theoretically, if reset_interval > 0, efficient seek() could
        be implemented. However, it isn't.

        Defaults to 0.
    """

    def __init__(self, sourcestream, window_bits=21, reset_interval=0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.sourcestream = sourcestream
        self.window_bits = window_bits
        self.reset_interval = reset_interval
        self.pos = None
        self.buf = None
        self.reset()

    def reset(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Resets the decompressor back to the start of the file.\n        '
        self.sourcestream.seek(0)
        self.decompressor = LZXDecompressor(self.sourcestream.read, self.window_bits, self.reset_interval)
        self.pos = 0
        self.buf = ByteQueue()

    def read(self, size: int=-1) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        if size < 0:
            size = INF
        while len(self.buf) < size:
            data = self.decompressor.decompress_next_frame()
            if not data:
                return self.buf.popleft(len(self.buf))
            self.buf.append(data)
        return self.buf.popleft(size)

    def get_size(self) -> int:
        if False:
            i = 10
            return i + 15
        del self
        return -1

    def seek(self, offset: int, whence=os.SEEK_SET) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        del offset, whence
        raise UnsupportedOperation('Cannot seek in LZXDStream.')

    def seekable(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def tell(self) -> int:
        if False:
            print('Hello World!')
        return self.pos

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.closed = True
        del self.decompressor
        del self.sourcestream