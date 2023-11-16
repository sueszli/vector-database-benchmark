"""This test checks for correct fork() behavior.
"""
import _imp as imp
import os
import signal
import sys
import threading
import time
import unittest
from test.fork_wait import ForkWait
from test import support
support.get_attribute(os, 'fork')

class ForkTest(ForkWait):

    def test_threaded_import_lock_fork(self):
        if False:
            i = 10
            return i + 15
        'Check fork() in main thread works while a subthread is doing an import'
        import_started = threading.Event()
        fake_module_name = 'fake test module'
        partial_module = 'partial'
        complete_module = 'complete'

        def importer():
            if False:
                for i in range(10):
                    print('nop')
            imp.acquire_lock()
            sys.modules[fake_module_name] = partial_module
            import_started.set()
            time.sleep(0.01)
            sys.modules[fake_module_name] = complete_module
            imp.release_lock()
        t = threading.Thread(target=importer)
        t.start()
        import_started.wait()
        exitcode = 42
        pid = os.fork()
        try:
            if not pid:
                m = __import__(fake_module_name)
                if m == complete_module:
                    os._exit(exitcode)
                else:
                    if support.verbose > 1:
                        print('Child encountered partial module')
                    os._exit(1)
            else:
                t.join()
                self.wait_impl(pid, exitcode=exitcode)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def test_nested_import_lock_fork(self):
        if False:
            for i in range(10):
                print('nop')
        'Check fork() in main thread works while the main thread is doing an import'
        exitcode = 42

        def fork_with_import_lock(level):
            if False:
                print('Hello World!')
            release = 0
            in_child = False
            try:
                try:
                    for i in range(level):
                        imp.acquire_lock()
                        release += 1
                    pid = os.fork()
                    in_child = not pid
                finally:
                    for i in range(release):
                        imp.release_lock()
            except RuntimeError:
                if in_child:
                    if support.verbose > 1:
                        print('RuntimeError in child')
                    os._exit(1)
                raise
            if in_child:
                os._exit(exitcode)
            self.wait_impl(pid, exitcode=exitcode)
        for level in range(5):
            fork_with_import_lock(level)

def tearDownModule():
    if False:
        while True:
            i = 10
    support.reap_children()
if __name__ == '__main__':
    unittest.main()