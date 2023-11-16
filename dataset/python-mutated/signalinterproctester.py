import os
import signal
import subprocess
import sys
import time
import unittest
from test import support

class SIGUSR1Exception(Exception):
    pass

class InterProcessSignalTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.got_signals = {'SIGHUP': 0, 'SIGUSR1': 0, 'SIGALRM': 0}

    def sighup_handler(self, signum, frame):
        if False:
            return 10
        self.got_signals['SIGHUP'] += 1

    def sigusr1_handler(self, signum, frame):
        if False:
            print('Hello World!')
        self.got_signals['SIGUSR1'] += 1
        raise SIGUSR1Exception

    def wait_signal(self, child, signame):
        if False:
            for i in range(10):
                print('nop')
        if child is not None:
            child.wait()
        timeout = support.SHORT_TIMEOUT
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.got_signals[signame]:
                return
            signal.pause()
        self.fail('signal %s not received after %s seconds' % (signame, timeout))

    def subprocess_send_signal(self, pid, signame):
        if False:
            print('Hello World!')
        code = 'import os, signal; os.kill(%s, signal.%s)' % (pid, signame)
        args = [sys.executable, '-I', '-c', code]
        return subprocess.Popen(args)

    def test_interprocess_signal(self):
        if False:
            return 10
        signal.signal(signal.SIGHUP, self.sighup_handler)
        signal.signal(signal.SIGUSR1, self.sigusr1_handler)
        signal.signal(signal.SIGUSR2, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, signal.default_int_handler)
        pid = str(os.getpid())
        with self.subprocess_send_signal(pid, 'SIGHUP') as child:
            self.wait_signal(child, 'SIGHUP')
        self.assertEqual(self.got_signals, {'SIGHUP': 1, 'SIGUSR1': 0, 'SIGALRM': 0})
        with self.assertRaises(SIGUSR1Exception):
            with self.subprocess_send_signal(pid, 'SIGUSR1') as child:
                self.wait_signal(child, 'SIGUSR1')
        self.assertEqual(self.got_signals, {'SIGHUP': 1, 'SIGUSR1': 1, 'SIGALRM': 0})
        with self.subprocess_send_signal(pid, 'SIGUSR2') as child:
            child.wait()
        try:
            with self.assertRaises(KeyboardInterrupt):
                signal.alarm(1)
                self.wait_signal(None, 'SIGALRM')
            self.assertEqual(self.got_signals, {'SIGHUP': 1, 'SIGUSR1': 1, 'SIGALRM': 0})
        finally:
            signal.alarm(0)
if __name__ == '__main__':
    unittest.main()