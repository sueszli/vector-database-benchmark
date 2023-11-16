"""A class to log stuff to a file, mainly useful in parallel situations."""
import datetime
import os

class FileLogger(object):
    """A logger to print stuff to a file."""

    def __init__(self, path, name, quiet=False, also_to_stdout=False):
        if False:
            return 10
        self._fd = open(os.path.join(path, 'log-{}.txt'.format(name)), 'w')
        self._quiet = quiet
        self.also_to_stdout = also_to_stdout

    def print(self, *args):
        if False:
            while True:
                i = 10
        date_prefix = '[{}]'.format(datetime.datetime.now().isoformat(' ')[:-3])
        print(date_prefix, *args, file=self._fd, flush=True)
        if self.also_to_stdout:
            print(date_prefix, *args, flush=True)

    def opt_print(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if not self._quiet:
            self.print(*args)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, unused_exception_type, unused_exc_value, unused_traceback):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def close(self):
        if False:
            i = 10
            return i + 15
        if self._fd:
            self._fd.close()
            self._fd = None

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self.close()