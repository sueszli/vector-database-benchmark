"""
Tests for L{twisted.internet._sigchld}, an alternate, superior SIGCHLD
monitoring API.
"""
from __future__ import annotations
import errno
import os
import signal
from twisted.python.runtime import platformType
from twisted.trial.unittest import SynchronousTestCase
if platformType == 'posix':
    from twisted.internet._signals import installHandler, isDefaultHandler
    from twisted.internet.fdesc import setNonBlocking
else:
    skip = 'These tests can only run on POSIX platforms.'

class SetWakeupSIGCHLDTests(SynchronousTestCase):
    """
    Tests for the L{signal.set_wakeup_fd} implementation of the
    L{installHandler} and L{isDefaultHandler} APIs.
    """

    def pipe(self) -> tuple[int, int]:
        if False:
            return 10
        '\n        Create a non-blocking pipe which will be closed after the currently\n        running test.\n        '
        (read, write) = os.pipe()
        self.addCleanup(os.close, read)
        self.addCleanup(os.close, write)
        setNonBlocking(read)
        setNonBlocking(write)
        return (read, write)

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Save the current SIGCHLD handler as reported by L{signal.signal} and\n        the current file descriptor registered with L{installHandler}.\n        '
        self.signalModuleHandler = signal.getsignal(signal.SIGCHLD)
        self.oldFD = installHandler(-1)

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Restore whatever signal handler was present when setUp ran.\n        '
        installHandler(self.oldFD)
        signal.signal(signal.SIGCHLD, self.signalModuleHandler)

    def test_isDefaultHandler(self) -> None:
        if False:
            return 10
        '\n        L{isDefaultHandler} returns true if the SIGCHLD handler is SIG_DFL,\n        false otherwise.\n        '
        self.assertTrue(isDefaultHandler())
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        self.assertFalse(isDefaultHandler())
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        self.assertTrue(isDefaultHandler())
        signal.signal(signal.SIGCHLD, lambda *args: None)
        self.assertFalse(isDefaultHandler())

    def test_returnOldFD(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{installHandler} returns the previously registered file descriptor.\n        '
        (read, write) = self.pipe()
        oldFD = installHandler(write)
        self.assertEqual(installHandler(oldFD), write)

    def test_uninstallHandler(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        C{installHandler(-1)} removes the SIGCHLD handler completely.\n        '
        (read, write) = self.pipe()
        self.assertTrue(isDefaultHandler())
        installHandler(write)
        self.assertFalse(isDefaultHandler())
        installHandler(-1)
        self.assertTrue(isDefaultHandler())

    def test_installHandler(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The file descriptor passed to L{installHandler} has a byte written to\n        it when SIGCHLD is delivered to the process.\n        '
        (read, write) = self.pipe()
        installHandler(write)
        exc = self.assertRaises(OSError, os.read, read, 1)
        self.assertEqual(exc.errno, errno.EAGAIN)
        os.kill(os.getpid(), signal.SIGCHLD)
        self.assertEqual(len(os.read(read, 5)), 1)