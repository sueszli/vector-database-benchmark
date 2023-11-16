from __future__ import annotations
import logging
from tempfile import mkdtemp
from virtualenv.util.path import safe_delete
from .via_disk_folder import AppDataDiskFolder

class TempAppData(AppDataDiskFolder):
    transient = True
    can_update = False

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(folder=mkdtemp())
        logging.debug('created temporary app data folder %s', self.lock.path)

    def reset(self):
        if False:
            print('Hello World!')
        'This is a temporary folder, is already empty to start with.'

    def close(self):
        if False:
            while True:
                i = 10
        logging.debug('remove temporary app data folder %s', self.lock.path)
        safe_delete(self.lock.path)

    def embed_update_log(self, distribution, for_py_version):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError
__all__ = ['TempAppData']