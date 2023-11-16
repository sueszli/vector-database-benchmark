import os
from threading import Timer

class wait_until_library:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._timers = []

    def remove_after_sleeping(self, *paths):
        if False:
            print('Hello World!')
        for p in paths:
            remover = os.rmdir if os.path.isdir(p) else os.remove
            self._run_after_sleeping(remover, p)

    def create_file_after_sleeping(self, path):
        if False:
            while True:
                i = 10
        self._run_after_sleeping(lambda : open(path, 'w').close())

    def create_dir_after_sleeping(self, path):
        if False:
            i = 10
            return i + 15
        self._run_after_sleeping(os.mkdir, path)

    def _run_after_sleeping(self, method, *args):
        if False:
            i = 10
            return i + 15
        self._timers.append(Timer(0.2, method, args))
        self._timers[-1].start()