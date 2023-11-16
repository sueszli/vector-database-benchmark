from __future__ import annotations
import mmap
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable
if TYPE_CHECKING:
    from http.client import HTTPResponse

class CallbackFileWrapper:
    """
    Small wrapper around a fp object which will tee everything read into a
    buffer, and when that file is closed it will execute a callback with the
    contents of that buffer.

    All attributes are proxied to the underlying file object.

    This class uses members with a double underscore (__) leading prefix so as
    not to accidentally shadow an attribute.

    The data is stored in a temporary file until it is all available.  As long
    as the temporary files directory is disk-based (sometimes it's a
    memory-backed-``tmpfs`` on Linux), data will be unloaded to disk if memory
    pressure is high.  For small files the disk usually won't be used at all,
    it'll all be in the filesystem memory cache, so there should be no
    performance impact.
    """

    def __init__(self, fp: HTTPResponse, callback: Callable[[bytes], None] | None) -> None:
        if False:
            i = 10
            return i + 15
        self.__buf = NamedTemporaryFile('rb+', delete=True)
        self.__fp = fp
        self.__callback = callback

    def __getattr__(self, name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        fp = self.__getattribute__('_CallbackFileWrapper__fp')
        return getattr(fp, name)

    def __is_fp_closed(self) -> bool:
        if False:
            return 10
        try:
            return self.__fp.fp is None
        except AttributeError:
            pass
        try:
            closed: bool = self.__fp.closed
            return closed
        except AttributeError:
            pass
        return False

    def _close(self) -> None:
        if False:
            print('Hello World!')
        if self.__callback:
            if self.__buf.tell() == 0:
                result = b''
            else:
                self.__buf.seek(0, 0)
                result = memoryview(mmap.mmap(self.__buf.fileno(), 0, access=mmap.ACCESS_READ))
            self.__callback(result)
        self.__callback = None
        self.__buf.close()

    def read(self, amt: int | None=None) -> bytes:
        if False:
            while True:
                i = 10
        data: bytes = self.__fp.read(amt)
        if data:
            self.__buf.write(data)
        if self.__is_fp_closed():
            self._close()
        return data

    def _safe_read(self, amt: int) -> bytes:
        if False:
            while True:
                i = 10
        data: bytes = self.__fp._safe_read(amt)
        if amt == 2 and data == b'\r\n':
            return data
        self.__buf.write(data)
        if self.__is_fp_closed():
            self._close()
        return data