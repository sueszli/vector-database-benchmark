"""
Provides FileLikeObject for binary stream interaction.
"""
from ..math import INF, clamp
from .readonly import PosSavingReadOnlyFileLikeObject
from ..bytequeue import ByteBuffer

class StreamSeekBuffer(PosSavingReadOnlyFileLikeObject):
    """
    Wrapper file-like object that adds seek functionality to a read-only,
    unseekable stream.

    For this purpose, all data read from that stream is cached.

    Constructor arguments:

    @param wrappee:
        The non-seekable, read-only stream that is to be wrapped.

    @param keepbuffered:
        If given, only this amount of bytes is guaranteed to stay in the
        seekback buffer.
        If too large of a seekback is requested, an attept at wrappee.reset()
        is made; if that doesn't work, tough luck.

    @param minread:
        If given, read calls to wrappee will request at least this amount
        of bytes (performance optimization).
        By default, entire megabytes are read at once.
    """

    def __init__(self, wrappee, keepbuffered: int=INF, minread: int=1048576):
        if False:
            while True:
                i = 10
        super().__init__()
        self.wrapped = wrappee
        self.keepbuffered = keepbuffered
        self.minread = minread
        self.buf = ByteBuffer()

    def resetwrappeed(self) -> None:
        if False:
            print('Hello World!')
        '\n        resets the wrappeed object, and clears self.buf.\n        '
        self.wrapped.reset()
        self.buf = ByteBuffer()

    def read(self, size: int=-1) -> bytes:
        if False:
            while True:
                i = 10
        if size < 0:
            size = INF
        if self.buf.hasbeendiscarded(self.pos):
            self.wrapped.reset()
            self.buf = ByteBuffer()
        needed = self.pos + size - len(self.buf)
        while needed > 0:
            amount = clamp(needed, self.minread, 67108864)
            data = self.wrapped.read(amount)
            self.buf.append(data)
            needed -= len(data)
            self.buf.discardleft(max(self.keepbuffered, size - needed))
            if len(data) < amount:
                break
        data = self.buf[self.pos:self.pos + size]
        self.pos += len(data)
        return data

    def get_size(self):
        if False:
            print('Hello World!')
        return self.wrapped.get_size()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.closed = True
        del self.buf
        del self.wrapped

class StreamFragment(PosSavingReadOnlyFileLikeObject):
    """
    Represents a definite part of an other file-like, read-only seekable
    stream.

    Constructor arguments:

    @param stream
        The stream; must implement read(), and seek() with whence=os.SEEK_SET.

        The stream's cursor is explicitly positioned before each call to
        read(); this allows multiple PartialStream objects to use the stream
        in parallel.

    @param start
        The first position of the stream that is used in this object.

    @param size
        The size of the stream fragment (in bytes).
    """

    def __init__(self, stream, start, size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.stream = stream
        self.start = start
        self.size = size
        if size < 0:
            raise ValueError('size must be positive')

    def read(self, size: int=-1) -> None:
        if False:
            i = 10
            return i + 15
        if size < 0:
            size = INF
        size = clamp(size, 0, self.size - self.pos)
        if not size:
            return b''
        self.stream.seek(self.start + self.pos)
        data = self.stream.read(size)
        if len(data) != size:
            raise EOFError('unexpected EOF in stream when attempting to read stream fragment')
        self.pos += len(data)
        return data

    def get_size(self) -> int:
        if False:
            while True:
                i = 10
        return self.size

    def close(self) -> None:
        if False:
            return 10
        self.closed = True
        del self.stream