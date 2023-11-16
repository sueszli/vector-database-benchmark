"""
Test support functionality for other tests.
"""
from __future__ import absolute_import
import time

class StopWatch(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.start_time = None
        self.duration = None

    def start(self):
        if False:
            return 10
        self.start_time = self.now()
        self.duration = None

    def stop(self):
        if False:
            while True:
                i = 10
        self.duration = self.now() - self.start_time

    @staticmethod
    def now():
        if False:
            i = 10
            return i + 15
        return time.time()

    @property
    def elapsed(self):
        if False:
            print('Hello World!')
        assert self.start_time is not None
        return self.now() - self.start_time

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        self.stop()