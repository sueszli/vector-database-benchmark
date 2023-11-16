import os
import time

class FileBaton:
    """A primitive, file-based synchronization utility."""

    def __init__(self, lock_file_path, wait_seconds=0.1):
        if False:
            return 10
        '\n        Create a new :class:`FileBaton`.\n\n        Args:\n            lock_file_path: The path to the file used for locking.\n            wait_seconds: The seconds to periodically sleep (spin) when\n                calling ``wait()``.\n        '
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None

    def try_acquire(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Try to atomically create a file under exclusive access.\n\n        Returns:\n            True if the file could be created, else False.\n        '
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):
        if False:
            return 10
        '\n        Periodically sleeps for a certain amount until the baton is released.\n\n        The amount of time slept depends on the ``wait_seconds`` parameter\n        passed to the constructor.\n        '
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        'Release the baton and removes its file.'
        if self.fd is not None:
            os.close(self.fd)
        os.remove(self.lock_file_path)