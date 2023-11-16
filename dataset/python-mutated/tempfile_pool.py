from __future__ import annotations
import atexit
import gc
import os
import tempfile
from types import TracebackType
from typing import Any
from typing import IO

class NamedTemporaryFilePool:
    tempfile_pool: list[IO[Any]] = []

    def __new__(cls, **kwargs: Any) -> 'NamedTemporaryFilePool':
        if False:
            return 10
        if not hasattr(cls, '_instance'):
            cls._instance = super(NamedTemporaryFilePool, cls).__new__(cls)
            atexit.register(cls._instance.cleanup)
        return cls._instance

    def __init__(self, **kwargs: Any) -> None:
        if False:
            return 10
        self.kwargs = kwargs

    def tempfile(self) -> IO[Any]:
        if False:
            return 10
        self._tempfile = tempfile.NamedTemporaryFile(delete=False, **self.kwargs)
        self.tempfile_pool.append(self._tempfile)
        return self._tempfile

    def cleanup(self) -> None:
        if False:
            i = 10
            return i + 15
        gc.collect()
        for i in self.tempfile_pool:
            os.unlink(i.name)

    def __enter__(self) -> IO[Any]:
        if False:
            while True:
                i = 10
        return self.tempfile()

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        if False:
            while True:
                i = 10
        self._tempfile.close()