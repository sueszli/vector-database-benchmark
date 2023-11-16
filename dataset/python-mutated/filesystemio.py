"""Utilities for ``FileSystem`` implementations."""
import abc
import io
import os
__all__ = ['Downloader', 'Uploader', 'DownloaderStream', 'UploaderStream', 'PipeStream']

class Downloader(metaclass=abc.ABCMeta):
    """Download interface for a single file.

  Implementations should support random access reads.
  """

    @property
    @abc.abstractmethod
    def size(self):
        if False:
            print('Hello World!')
        'Size of file to download.'

    @abc.abstractmethod
    def get_range(self, start, end):
        if False:
            print('Hello World!')
        'Retrieve a given byte range [start, end) from this download.\n\n    Range must be in this form:\n      0 <= start < end: Fetch the bytes from start to end.\n\n    Args:\n      start: (int) Initial byte offset.\n      end: (int) Final byte offset, exclusive.\n\n    Returns:\n      (string) A buffer containing the requested data.\n    '

class Uploader(metaclass=abc.ABCMeta):
    """Upload interface for a single file."""

    @abc.abstractmethod
    def put(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Write data to file sequentially.\n\n    Args:\n      data: (memoryview) Data to write.\n    '

    @abc.abstractmethod
    def finish(self):
        if False:
            i = 10
            return i + 15
        'Signal to upload any remaining data and close the file.\n\n    File should be fully written upon return from this method.\n\n    Raises:\n      Any error encountered during the upload.\n    '

class DownloaderStream(io.RawIOBase):
    """Provides a stream interface for Downloader objects."""

    def __init__(self, downloader, read_buffer_size=io.DEFAULT_BUFFER_SIZE, mode='rb'):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the stream.\n\n    Args:\n      downloader: (Downloader) Filesystem dependent implementation.\n      read_buffer_size: (int) Buffer size to use during read operations.\n      mode: (string) Python mode attribute for this stream.\n    '
        self._downloader = downloader
        self.mode = mode
        self._position = 0
        self._reader_buffer_size = read_buffer_size

    def readinto(self, b):
        if False:
            return 10
        'Read up to len(b) bytes into b.\n\n    Returns number of bytes read (0 for EOF).\n\n    Args:\n      b: (bytearray/memoryview) Buffer to read into.\n    '
        self._checkClosed()
        if self._position >= self._downloader.size:
            return 0
        start = self._position
        end = min(self._position + len(b), self._downloader.size)
        data = self._downloader.get_range(start, end)
        self._position += len(data)
        b[:len(data)] = data
        return len(data)

    def seek(self, offset, whence=os.SEEK_SET):
        if False:
            while True:
                i = 10
        "Set the stream's current offset.\n\n    Note if the new offset is out of bound, it is adjusted to either 0 or EOF.\n\n    Args:\n      offset: seek offset as number.\n      whence: seek mode. Supported modes are os.SEEK_SET (absolute seek),\n        os.SEEK_CUR (seek relative to the current position), and os.SEEK_END\n        (seek relative to the end, offset should be negative).\n\n    Raises:\n      ``ValueError``: When this stream is closed or if whence is invalid.\n    "
        self._checkClosed()
        if whence == os.SEEK_SET:
            self._position = offset
        elif whence == os.SEEK_CUR:
            self._position += offset
        elif whence == os.SEEK_END:
            self._position = self._downloader.size + offset
        else:
            raise ValueError('Whence mode %r is invalid.' % whence)
        self._position = min(self._position, self._downloader.size)
        self._position = max(self._position, 0)
        return self._position

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        "Tell the stream's current offset.\n\n    Returns:\n      current offset in reading this stream.\n\n    Raises:\n      ``ValueError``: When this stream is closed.\n    "
        self._checkClosed()
        return self._position

    def seekable(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def readable(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def readall(self):
        if False:
            i = 10
            return i + 15
        'Read until EOF, using multiple read() call.'
        res = []
        while True:
            data = self.read(self._reader_buffer_size)
            if not data:
                break
            res.append(data)
        return b''.join(res)

class UploaderStream(io.RawIOBase):
    """Provides a stream interface for Uploader objects."""

    def __init__(self, uploader, mode='wb'):
        if False:
            while True:
                i = 10
        'Initializes the stream.\n\n    Args:\n      uploader: (Uploader) Filesystem dependent implementation.\n      mode: (string) Python mode attribute for this stream.\n    '
        self._uploader = uploader
        self.mode = mode
        self._position = 0

    def tell(self):
        if False:
            print('Hello World!')
        return self._position

    def write(self, b):
        if False:
            i = 10
            return i + 15
        'Write bytes from b.\n\n    Returns number of bytes written (<= len(b)).\n\n    Args:\n      b: (memoryview) Buffer with data to write.\n    '
        self._checkClosed()
        self._uploader.put(b)
        bytes_written = len(b)
        self._position += bytes_written
        return bytes_written

    def close(self):
        if False:
            print('Hello World!')
        'Complete the upload and close this stream.\n\n    This method has no effect if the stream is already closed.\n\n    Raises:\n      Any error encountered by the uploader.\n    '
        if not self.closed:
            self._uploader.finish()
        super().close()

    def writable(self):
        if False:
            return 10
        return True

class PipeStream(object):
    """A class that presents a pipe connection as a readable stream.

  Not thread-safe.

  Remembers the last ``size`` bytes read and allows rewinding the stream by that
  amount exactly. See BEAM-6380 for more.
  """

    def __init__(self, recv_pipe):
        if False:
            while True:
                i = 10
        self.conn = recv_pipe
        self.closed = False
        self.position = 0
        self.remaining = b''
        self.last_block_position = None
        self.last_block = b''

    def read(self, size):
        if False:
            i = 10
            return i + 15
        'Read data from the wrapped pipe connection.\n\n    Args:\n      size: Number of bytes to read. Actual number of bytes read is always\n            equal to size unless EOF is reached.\n\n    Returns:\n      data read as str.\n    '
        data_list = []
        bytes_read = 0
        last_block_position = self.position
        while bytes_read < size:
            bytes_from_remaining = min(size - bytes_read, len(self.remaining))
            data_list.append(self.remaining[0:bytes_from_remaining])
            self.remaining = self.remaining[bytes_from_remaining:]
            self.position += bytes_from_remaining
            bytes_read += bytes_from_remaining
            if not self.remaining:
                try:
                    self.remaining = self.conn.recv_bytes()
                except EOFError:
                    break
        last_block = b''.join(data_list)
        if last_block:
            self.last_block_position = last_block_position
            self.last_block = last_block
        return last_block

    def tell(self):
        if False:
            print('Hello World!')
        "Tell the file's current offset.\n\n    Returns:\n      current offset in reading this file.\n\n    Raises:\n      ``ValueError``: When this stream is closed.\n    "
        self._check_open()
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if False:
            return 10
        if whence == os.SEEK_END and offset == 0:
            return
        elif whence == os.SEEK_SET:
            if offset == self.position:
                return
            elif offset == self.last_block_position and self.last_block:
                self.position = offset
                self.remaining = b''.join([self.last_block, self.remaining])
                self.last_block = b''
                return
        raise NotImplementedError('offset: %s, whence: %s, position: %s, last: %s' % (offset, whence, self.position, self.last_block_position))

    def _check_open(self):
        if False:
            i = 10
            return i + 15
        if self.closed:
            raise IOError('Stream is closed.')