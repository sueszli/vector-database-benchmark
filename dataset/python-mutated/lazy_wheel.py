"""Lazy ZIP over HTTP"""
__all__ = ['HTTPRangeRequestUnsupported', 'dist_from_wheel_url']
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.metadata import BaseDistribution, MemoryWheel, get_wheel_distribution
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks

class HTTPRangeRequestUnsupported(Exception):
    pass

def dist_from_wheel_url(name: str, url: str, session: PipSession) -> BaseDistribution:
    if False:
        i = 10
        return i + 15
    'Return a distribution object from the given wheel URL.\n\n    This uses HTTP range requests to only fetch the portion of the wheel\n    containing metadata, just enough for the object to be constructed.\n    If such requests are not supported, HTTPRangeRequestUnsupported\n    is raised.\n    '
    with LazyZipOverHTTP(url, session) as zf:
        wheel = MemoryWheel(zf.name, zf)
        return get_wheel_distribution(wheel, canonicalize_name(name))

class LazyZipOverHTTP:
    """File-like object mapped to a ZIP file over HTTP.

    This uses HTTP range requests to lazily fetch the file's content,
    which is supposed to be fed to ZipFile.  If such requests are not
    supported by the server, raise HTTPRangeRequestUnsupported
    during initialization.
    """

    def __init__(self, url: str, session: PipSession, chunk_size: int=CONTENT_CHUNK_SIZE) -> None:
        if False:
            while True:
                i = 10
        head = session.head(url, headers=HEADERS)
        raise_for_status(head)
        assert head.status_code == 200
        (self._session, self._url, self._chunk_size) = (session, url, chunk_size)
        self._length = int(head.headers['Content-Length'])
        self._file = NamedTemporaryFile()
        self.truncate(self._length)
        self._left: List[int] = []
        self._right: List[int] = []
        if 'bytes' not in head.headers.get('Accept-Ranges', 'none'):
            raise HTTPRangeRequestUnsupported('range request is not supported')
        self._check_zip()

    @property
    def mode(self) -> str:
        if False:
            print('Hello World!')
        'Opening mode, which is always rb.'
        return 'rb'

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Path to the underlying file.'
        return self._file.name

    def seekable(self) -> bool:
        if False:
            print('Hello World!')
        'Return whether random access is supported, which is True.'
        return True

    def close(self) -> None:
        if False:
            return 10
        'Close the file.'
        self._file.close()

    @property
    def closed(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether the file is closed.'
        return self._file.closed

    def read(self, size: int=-1) -> bytes:
        if False:
            return 10
        'Read up to size bytes from the object and return them.\n\n        As a convenience, if size is unspecified or -1,\n        all bytes until EOF are returned.  Fewer than\n        size bytes may be returned if EOF is reached.\n        '
        download_size = max(size, self._chunk_size)
        (start, length) = (self.tell(), self._length)
        stop = length if size < 0 else min(start + download_size, length)
        start = max(0, stop - download_size)
        self._download(start, stop - 1)
        return self._file.read(size)

    def readable(self) -> bool:
        if False:
            print('Hello World!')
        'Return whether the file is readable, which is True.'
        return True

    def seek(self, offset: int, whence: int=0) -> int:
        if False:
            while True:
                i = 10
        'Change stream position and return the new absolute position.\n\n        Seek to offset relative position indicated by whence:\n        * 0: Start of stream (the default).  pos should be >= 0;\n        * 1: Current position - pos may be negative;\n        * 2: End of stream - pos usually negative.\n        '
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        if False:
            return 10
        'Return the current position.'
        return self._file.tell()

    def truncate(self, size: Optional[int]=None) -> int:
        if False:
            print('Hello World!')
        "Resize the stream to the given size in bytes.\n\n        If size is unspecified resize to the current position.\n        The current stream position isn't changed.\n\n        Return the new file size.\n        "
        return self._file.truncate(size)

    def writable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return False.'
        return False

    def __enter__(self) -> 'LazyZipOverHTTP':
        if False:
            return 10
        self._file.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if False:
            print('Hello World!')
        self._file.__exit__(*exc)

    @contextmanager
    def _stay(self) -> Generator[None, None, None]:
        if False:
            i = 10
            return i + 15
        'Return a context manager keeping the position.\n\n        At the end of the block, seek back to original position.\n        '
        pos = self.tell()
        try:
            yield
        finally:
            self.seek(pos)

    def _check_zip(self) -> None:
        if False:
            while True:
                i = 10
        'Check and download until the file is a valid ZIP.'
        end = self._length - 1
        for start in reversed(range(0, end, self._chunk_size)):
            self._download(start, end)
            with self._stay():
                try:
                    ZipFile(self)
                except BadZipFile:
                    pass
                else:
                    break

    def _stream_response(self, start: int, end: int, base_headers: Dict[str, str]=HEADERS) -> Response:
        if False:
            while True:
                i = 10
        'Return HTTP response to a range request from start to end.'
        headers = base_headers.copy()
        headers['Range'] = f'bytes={start}-{end}'
        headers['Cache-Control'] = 'no-cache'
        return self._session.get(self._url, headers=headers, stream=True)

    def _merge(self, start: int, end: int, left: int, right: int) -> Generator[Tuple[int, int], None, None]:
        if False:
            return 10
        'Return a generator of intervals to be fetched.\n\n        Args:\n            start (int): Start of needed interval\n            end (int): End of needed interval\n            left (int): Index of first overlapping downloaded data\n            right (int): Index after last overlapping downloaded data\n        '
        (lslice, rslice) = (self._left[left:right], self._right[left:right])
        i = start = min([start] + lslice[:1])
        end = max([end] + rslice[-1:])
        for (j, k) in zip(lslice, rslice):
            if j > i:
                yield (i, j - 1)
            i = k + 1
        if i <= end:
            yield (i, end)
        (self._left[left:right], self._right[left:right]) = ([start], [end])

    def _download(self, start: int, end: int) -> None:
        if False:
            print('Hello World!')
        'Download bytes from start to end inclusively.'
        with self._stay():
            left = bisect_left(self._right, start)
            right = bisect_right(self._left, end)
            for (start, end) in self._merge(start, end, left, right):
                response = self._stream_response(start, end)
                response.raise_for_status()
                self.seek(start)
                for chunk in response_chunks(response, self._chunk_size):
                    self._file.write(chunk)