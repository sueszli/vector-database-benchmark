from __future__ import print_function
import sys
import subprocess
import unittest
from gevent.thread import allocate_lock
import gevent.testing as greentest
script = '\nfrom gevent import monkey\nmonkey.patch_all() # pragma: testrunner-no-monkey-combine\nimport sys, os, threading, time\n\n\n# A deadlock-killer, to prevent the\n# testsuite to hang forever\ndef killer():\n    time.sleep(0.2)\n    sys.stdout.write(\'..program blocked; aborting!\')\n    sys.stdout.flush()\n    os._exit(2)\nt = threading.Thread(target=killer)\nt.daemon = True\nt.start()\n\n\ndef trace(frame, event, arg):\n    if threading is not None:\n        threading.current_thread()\n    return trace\n\n\ndef doit():\n    sys.stdout.write("..thread started..")\n\n\ndef test1():\n    t = threading.Thread(target=doit)\n    t.start()\n    t.join()\n    sys.settrace(None)\n\nsys.settrace(trace)\nif len(sys.argv) > 1:\n    test1()\n\nsys.stdout.write("..finishing..")\n'

class TestTrace(unittest.TestCase):

    @greentest.skipOnPurePython('Locks can be traced in Pure Python')
    def test_untraceable_lock(self):
        if False:
            while True:
                i = 10
        if hasattr(sys, 'gettrace'):
            old = sys.gettrace()
        else:
            old = None
        lst = []
        try:

            def trace(frame, ev, _arg):
                if False:
                    print('Hello World!')
                lst.append((frame.f_code.co_filename, frame.f_lineno, ev))
                print('TRACE: %s:%s %s' % lst[-1])
                return trace
            with allocate_lock():
                sys.settrace(trace)
        finally:
            sys.settrace(old)
        self.assertEqual(lst, [], 'trace not empty')

    @greentest.skipOnPurePython('Locks can be traced in Pure Python')
    def test_untraceable_lock_uses_different_lock(self):
        if False:
            return 10
        if hasattr(sys, 'gettrace'):
            old = sys.gettrace()
        else:
            old = None
        lst = []
        l = allocate_lock()
        try:

            def trace(frame, ev, _arg):
                if False:
                    i = 10
                    return i + 15
                with l:
                    lst.append((frame.f_code.co_filename, frame.f_lineno, ev))
                return trace
            l2 = allocate_lock()
            sys.settrace(trace)
            l2.acquire()
            l2.release()
        finally:
            sys.settrace(old)
        self.assertTrue(lst, 'should not compile on pypy')

    @greentest.skipOnPurePython('Locks can be traced in Pure Python')
    def test_untraceable_lock_uses_same_lock(self):
        if False:
            print('Hello World!')
        from gevent.hub import LoopExit
        if hasattr(sys, 'gettrace'):
            old = sys.gettrace()
        else:
            old = None
        lst = []
        e = None
        l = allocate_lock()
        try:

            def trace(frame, ev, _arg):
                if False:
                    return 10
                with l:
                    lst.append((frame.f_code.co_filename, frame.f_lineno, ev))
                return trace
            sys.settrace(trace)
            l.acquire()
        except LoopExit as ex:
            e = ex
        finally:
            sys.settrace(old)
        self.assertTrue(lst, 'should not compile on pypy')
        self.assertTrue(isinstance(e, LoopExit))

    def run_script(self, more_args=()):
        if False:
            for i in range(10):
                print('nop')
        if greentest.PYPY3 and greentest.RUNNING_ON_APPVEYOR and (sys.version_info[:2] == (3, 7)):
            self.skipTest('Known to hang on AppVeyor')
        args = [sys.executable, '-u', '-c', script]
        args.extend(more_args)
        rc = subprocess.call(args)
        self.assertNotEqual(rc, 2, 'interpreter was blocked')
        self.assertEqual(rc, 0, 'Unexpected error')

    def test_finalize_with_trace(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_script()

    def test_bootstrap_inner_with_trace(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_script(['1'])
if __name__ == '__main__':
    greentest.main()