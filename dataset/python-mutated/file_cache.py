from __future__ import annotations
import hashlib
import os
from textwrap import dedent
from typing import IO, TYPE_CHECKING
from pip._vendor.cachecontrol.cache import BaseCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.controller import CacheController
if TYPE_CHECKING:
    from datetime import datetime
    from filelock import BaseFileLock

def _secure_open_write(filename: str, fmode: int) -> IO[bytes]:
    if False:
        return 10
    flags = os.O_WRONLY
    flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    if hasattr(os, 'O_BINARY'):
        flags |= os.O_BINARY
    try:
        os.remove(filename)
    except OSError:
        pass
    fd = os.open(filename, flags, fmode)
    try:
        return os.fdopen(fd, 'wb')
    except:
        os.close(fd)
        raise

class _FileCacheMixin:
    """Shared implementation for both FileCache variants."""

    def __init__(self, directory: str, forever: bool=False, filemode: int=384, dirmode: int=448, lock_class: type[BaseFileLock] | None=None) -> None:
        if False:
            while True:
                i = 10
        try:
            if lock_class is None:
                from filelock import FileLock
                lock_class = FileLock
        except ImportError:
            notice = dedent('\n            NOTE: In order to use the FileCache you must have\n            filelock installed. You can install it via pip:\n              pip install filelock\n            ')
            raise ImportError(notice)
        self.directory = directory
        self.forever = forever
        self.filemode = filemode
        self.dirmode = dirmode
        self.lock_class = lock_class

    @staticmethod
    def encode(x: str) -> str:
        if False:
            print('Hello World!')
        return hashlib.sha224(x.encode()).hexdigest()

    def _fn(self, name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        hashed = self.encode(name)
        parts = list(hashed[:5]) + [hashed]
        return os.path.join(self.directory, *parts)

    def get(self, key: str) -> bytes | None:
        if False:
            return 10
        name = self._fn(key)
        try:
            with open(name, 'rb') as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def set(self, key: str, value: bytes, expires: int | datetime | None=None) -> None:
        if False:
            print('Hello World!')
        name = self._fn(key)
        self._write(name, value)

    def _write(self, path: str, data: bytes) -> None:
        if False:
            print('Hello World!')
        '\n        Safely write the data to the given path.\n        '
        try:
            os.makedirs(os.path.dirname(path), self.dirmode)
        except OSError:
            pass
        with self.lock_class(path + '.lock'):
            with _secure_open_write(path, self.filemode) as fh:
                fh.write(data)

    def _delete(self, key: str, suffix: str) -> None:
        if False:
            print('Hello World!')
        name = self._fn(key) + suffix
        if not self.forever:
            try:
                os.remove(name)
            except FileNotFoundError:
                pass

class FileCache(_FileCacheMixin, BaseCache):
    """
    Traditional FileCache: body is stored in memory, so not suitable for large
    downloads.
    """

    def delete(self, key: str) -> None:
        if False:
            return 10
        self._delete(key, '')

class SeparateBodyFileCache(_FileCacheMixin, SeparateBodyBaseCache):
    """
    Memory-efficient FileCache: body is stored in a separate file, reducing
    peak memory usage.
    """

    def get_body(self, key: str) -> IO[bytes] | None:
        if False:
            for i in range(10):
                print('nop')
        name = self._fn(key) + '.body'
        try:
            return open(name, 'rb')
        except FileNotFoundError:
            return None

    def set_body(self, key: str, body: bytes) -> None:
        if False:
            return 10
        name = self._fn(key) + '.body'
        self._write(name, body)

    def delete(self, key: str) -> None:
        if False:
            while True:
                i = 10
        self._delete(key, '')
        self._delete(key, '.body')

def url_to_file_path(url: str, filecache: FileCache) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Return the file cache path based on the URL.\n\n    This does not ensure the file exists!\n    '
    key = CacheController.cache_url(url)
    return filecache._fn(key)