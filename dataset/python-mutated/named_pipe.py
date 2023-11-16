import abc
import logging
import os
import random
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Type
from streamlink.compat import is_win32
try:
    from ctypes import byref, c_ulong, c_void_p, cast, windll
except ImportError:
    pass
log = logging.getLogger(__name__)
_lock = threading.Lock()
_id = 0

class NamedPipeBase(abc.ABC):
    path: Path

    def __init__(self):
        if False:
            print('Hello World!')
        global _id
        with _lock:
            _id += 1
            self.name = f'streamlinkpipe-{os.getpid()}-{_id}-{random.randint(0, 9999)}'
        log.info(f'Creating pipe {self.name}')
        self._create()

    @abc.abstractmethod
    def _create(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abc.abstractmethod
    def open(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, data) -> int:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

class NamedPipePosix(NamedPipeBase):
    mode = 'wb'
    permissions = 432
    fifo = None

    def _create(self):
        if False:
            while True:
                i = 10
        self.path = Path(tempfile.gettempdir(), self.name)
        os.mkfifo(self.path, self.permissions)

    def open(self):
        if False:
            return 10
        self.fifo = open(self.path, self.mode)

    def write(self, data):
        if False:
            while True:
                i = 10
        return self.fifo.write(data)

    def close(self):
        if False:
            while True:
                i = 10
        try:
            if self.fifo is not None:
                self.fifo.close()
        except OSError:
            raise
        finally:
            with suppress(OSError):
                self.path.unlink()
            self.fifo = None

class NamedPipeWindows(NamedPipeBase):
    bufsize = 8192
    pipe = None
    PIPE_ACCESS_OUTBOUND = 2
    PIPE_TYPE_BYTE = 0
    PIPE_READMODE_BYTE = 0
    PIPE_WAIT = 0
    PIPE_UNLIMITED_INSTANCES = 255
    INVALID_HANDLE_VALUE = -1

    @staticmethod
    def _get_last_error():
        if False:
            for i in range(10):
                print('nop')
        error_code = windll.kernel32.GetLastError()
        raise OSError(f'Named pipe error code 0x{error_code:08X}')

    def _create(self):
        if False:
            i = 10
            return i + 15
        self.path = Path('\\\\.\\pipe', self.name)
        self.pipe = windll.kernel32.CreateNamedPipeW(str(self.path), self.PIPE_ACCESS_OUTBOUND, self.PIPE_TYPE_BYTE | self.PIPE_READMODE_BYTE | self.PIPE_WAIT, self.PIPE_UNLIMITED_INSTANCES, self.bufsize, self.bufsize, 0, None)
        if self.pipe == self.INVALID_HANDLE_VALUE:
            self._get_last_error()

    def open(self):
        if False:
            while True:
                i = 10
        windll.kernel32.ConnectNamedPipe(self.pipe, None)

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        written = c_ulong(0)
        windll.kernel32.WriteFile(self.pipe, cast(data, c_void_p), len(data), byref(written), None)
        return written.value

    def close(self):
        if False:
            print('Hello World!')
        try:
            if self.pipe is not None:
                windll.kernel32.DisconnectNamedPipe(self.pipe)
                windll.kernel32.CloseHandle(self.pipe)
        except OSError:
            raise
        finally:
            self.pipe = None
NamedPipe: Type[NamedPipeBase]
if not is_win32:
    NamedPipe = NamedPipePosix
else:
    NamedPipe = NamedPipeWindows