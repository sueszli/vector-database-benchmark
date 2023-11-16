from __future__ import absolute_import, division, print_function, unicode_literals
import sys

class flushfile(object):

    def __init__(self, f):
        if False:
            for i in range(10):
                print('nop')
        self.f = f

    def write(self, x):
        if False:
            while True:
                i = 10
        self.f.write(x)
        self.f.flush()

    def flush(self):
        if False:
            while True:
                i = 10
        self.f.flush()
if sys.platform == 'win32':
    sys.stdout = flushfile(sys.stdout)
    sys.stderr = flushfile(sys.stderr)

class StdOutDevNull(object):

    def __init__(self):
        if False:
            return 10
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, x):
        if False:
            i = 10
            return i + 15
        pass

    def flush(self):
        if False:
            i = 10
            return i + 15
        pass

    def stop(self):
        if False:
            return 10
        sys.stdout = self.stdout