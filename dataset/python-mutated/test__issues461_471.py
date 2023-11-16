"""Test for GitHub issues 461 and 471.

When moving to Python 3, handling of KeyboardInterrupt exceptions caused
by a Ctrl-C raised an exception while printing the traceback for a
greenlet preventing the process from exiting. This test tests for proper
handling of KeyboardInterrupt.
"""
import sys
if sys.argv[1:] == ['subprocess']:
    import gevent

    def task():
        if False:
            i = 10
            return i + 15
        sys.stdout.write('ready\n')
        sys.stdout.flush()
        gevent.sleep(30)
    try:
        gevent.spawn(task).get()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
else:
    import signal
    from subprocess import Popen, PIPE
    import time
    import unittest
    import gevent.testing as greentest
    from gevent.testing.sysinfo import CFFI_BACKEND
    from gevent.testing.sysinfo import RUN_COVERAGE
    from gevent.testing.sysinfo import WIN
    from gevent.testing.sysinfo import PYPY3

    class Test(unittest.TestCase):

        @unittest.skipIf(CFFI_BACKEND and RUN_COVERAGE or (PYPY3 and WIN), 'Interferes with the timing; times out waiting for the child')
        def test_hang(self):
            if False:
                return 10
            if WIN:
                from subprocess import CREATE_NEW_PROCESS_GROUP
                kwargs = {'creationflags': CREATE_NEW_PROCESS_GROUP}
            else:
                kwargs = {}
            p = Popen([sys.executable, __file__, 'subprocess'], stdout=PIPE, **kwargs)
            line = p.stdout.readline()
            if not isinstance(line, str):
                line = line.decode('ascii')
            line = line.strip()
            self.assertEqual(line, 'ready')
            signal_to_send = signal.SIGINT if not WIN else getattr(signal, 'CTRL_BREAK_EVENT')
            p.send_signal(signal_to_send)
            wait_seconds = 25.0
            now = time.time()
            midtime = now + wait_seconds / 2.0
            endtime = time.time() + wait_seconds
            while time.time() < endtime:
                if p.poll() is not None:
                    break
                if time.time() > midtime:
                    p.send_signal(signal_to_send)
                    midtime = endtime + 1
                time.sleep(0.1)
            else:
                p.terminate()
                p.wait()
                raise AssertionError('Failed to wait for child')
            self.assertEqual(p.returncode if not WIN else 0, 0)
            p.stdout.close()
    if __name__ == '__main__':
        greentest.main()