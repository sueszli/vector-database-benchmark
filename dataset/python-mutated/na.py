from __future__ import annotations
from contextlib import contextmanager
from .base import AppData, ContentStore

class AppDataDisabled(AppData):
    """No application cache available (most likely as we don't have write permissions)."""
    transient = True
    can_update = False

    def __init__(self) -> None:
        if False:
            return 10
        pass
    error = RuntimeError('no app data folder available, probably no write access to the folder')

    def close(self):
        if False:
            i = 10
            return i + 15
        'Do nothing.'

    def reset(self):
        if False:
            return 10
        'Do nothing.'

    def py_info(self, path):
        if False:
            while True:
                i = 10
        return ContentStoreNA()

    def embed_update_log(self, distribution, for_py_version):
        if False:
            i = 10
            return i + 15
        return ContentStoreNA()

    def extract(self, path, to_folder):
        if False:
            return 10
        raise self.error

    @contextmanager
    def locked(self, path):
        if False:
            return 10
        'Do nothing.'
        yield

    @property
    def house(self):
        if False:
            while True:
                i = 10
        raise self.error

    def wheel_image(self, for_py_version, name):
        if False:
            while True:
                i = 10
        raise self.error

    def py_info_clear(self):
        if False:
            return 10
        'Nothing to clear.'

class ContentStoreNA(ContentStore):

    def exists(self):
        if False:
            return 10
        return False

    def read(self):
        if False:
            i = 10
            return i + 15
        'Nothing to read.'
        return

    def write(self, content):
        if False:
            print('Hello World!')
        'Nothing to write.'

    def remove(self):
        if False:
            for i in range(10):
                print('nop')
        'Nothing to remove.'

    @contextmanager
    def locked(self):
        if False:
            return 10
        yield
__all__ = ['AppDataDisabled', 'ContentStoreNA']