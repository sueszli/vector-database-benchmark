"""
Run all of the unit tests for this package multiple times in a highly
multi-threaded way to stress the system. This makes it possible to look
for memory leaks and threading issues and provides a good target for a
profiler to accumulate better data.
"""
from __future__ import print_function
import gc
import os
import sys
import threading
import time
import _thread as thread
from .utils import dprint

class StressTest(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.dirname = os.path.split(__file__)[0]
        sys.path.append(self.dirname)
        gc.set_debug(gc.DEBUG_LEAK)
        import runtests
        self.module = runtests
        self.done = []

    def mark_start(self):
        if False:
            for i in range(10):
                print('nop')
        self._start = time.clock()

    def mark_finish(self):
        if False:
            while True:
                i = 10
        self._finish = time.clock()

    def elapsed(self):
        if False:
            while True:
                i = 10
        return self._finish - self._start

    def print_gc_report(self):
        if False:
            i = 10
            return i + 15
        for item in gc.get_objects():
            print(item, sys.getrefcount(item))

    def run_thread(self, iterations):
        if False:
            for i in range(10):
                print('nop')
        thread_id = thread.get_ident()
        dprint('thread {0} starting...'.format(thread_id))
        time.sleep(0.1)
        for i in range(iterations):
            dprint('thread {0} iter {1} start'.format(thread_id, i))
            self.module.main()
            dprint('thread {0} iter {1} end'.format(thread_id, i))
        self.done.append(None)
        dprint('thread {0} done'.format(thread_id))

    def stress_test(self, iterations=1, threads=1):
        if False:
            i = 10
            return i + 15
        args = (iterations,)
        self.mark_start()
        for _ in range(threads):
            thread = threading.Thread(target=self.run_thread, args=args)
            thread.start()
        while len(self.done) < iterations * threads:
            dprint(len(self.done))
            time.sleep(0.1)
        self.mark_finish()
        took = self.elapsed()
        self.print_gc_report()

def main():
    if False:
        print('Hello World!')
    test = StressTest()
    test.stress_test(2, 10)
if __name__ == '__main__':
    main()