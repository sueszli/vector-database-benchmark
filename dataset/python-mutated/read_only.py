from __future__ import annotations
import os.path
from virtualenv.util.lock import NoOpFileLock
from .via_disk_folder import AppDataDiskFolder, PyInfoStoreDisk

class ReadOnlyAppData(AppDataDiskFolder):
    can_update = False

    def __init__(self, folder: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not os.path.isdir(folder):
            msg = f'read-only app data directory {folder} does not exist'
            raise RuntimeError(msg)
        super().__init__(folder)
        self.lock = NoOpFileLock(folder)

    def reset(self) -> None:
        if False:
            return 10
        msg = 'read-only app data does not support reset'
        raise RuntimeError(msg)

    def py_info_clear(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def py_info(self, path):
        if False:
            print('Hello World!')
        return _PyInfoStoreDiskReadOnly(self.py_info_at, path)

    def embed_update_log(self, distribution, for_py_version):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class _PyInfoStoreDiskReadOnly(PyInfoStoreDisk):

    def write(self, content):
        if False:
            return 10
        msg = 'read-only app data python info cannot be updated'
        raise RuntimeError(msg)
__all__ = ['ReadOnlyAppData']