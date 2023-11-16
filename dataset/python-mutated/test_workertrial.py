"""
Tests for L{twisted.trial._dist.workertrial}.
"""
import errno
import sys
from io import BytesIO
from twisted.internet.testing import StringTransport
from twisted.protocols.amp import AMP
from twisted.trial._dist import _WORKER_AMP_STDIN, _WORKER_AMP_STDOUT, managercommands, workercommands, workertrial
from twisted.trial._dist.workertrial import WorkerLogObserver, main
from twisted.trial.unittest import TestCase

class FakeAMP(AMP):
    """
    A fake amp protocol.
    """

class WorkerLogObserverTests(TestCase):
    """
    Tests for L{WorkerLogObserver}.
    """

    def test_emit(self):
        if False:
            while True:
                i = 10
        '\n        L{WorkerLogObserver} forwards data to L{managercommands.TestWrite}.\n        '
        calls = []

        class FakeClient:

            def callRemote(self, method, **kwargs):
                if False:
                    i = 10
                    return i + 15
                calls.append((method, kwargs))
        observer = WorkerLogObserver(FakeClient())
        observer.emit({'message': ['Some log']})
        self.assertEqual(calls, [(managercommands.TestWrite, {'out': 'Some log'})])

class MainTests(TestCase):
    """
    Tests for L{main}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.readStream = BytesIO()
        self.writeStream = BytesIO()
        self.patch(workertrial, 'startLoggingWithObserver', self.startLoggingWithObserver)
        self.addCleanup(setattr, sys, 'argv', sys.argv)
        sys.argv = ['trial']

    def fdopen(self, fd, mode=None):
        if False:
            return 10
        '\n        Fake C{os.fdopen} implementation which returns C{self.readStream} for\n        the stdin fd and C{self.writeStream} for the stdout fd.\n        '
        if fd == _WORKER_AMP_STDIN:
            self.assertEqual('rb', mode)
            return self.readStream
        elif fd == _WORKER_AMP_STDOUT:
            self.assertEqual('wb', mode)
            return self.writeStream
        else:
            raise AssertionError(f'Unexpected fd {fd!r}')

    def startLoggingWithObserver(self, emit, setStdout):
        if False:
            while True:
                i = 10
        '\n        Override C{startLoggingWithObserver} for not starting logging.\n        '
        self.assertFalse(setStdout)

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        '\n        If no data is ever written, L{main} exits without writing data out.\n        '
        main(self.fdopen)
        self.assertEqual(b'', self.writeStream.getvalue())

    def test_forwardCommand(self):
        if False:
            return 10
        '\n        L{main} forwards data from its input stream to a L{WorkerProtocol}\n        instance which writes data to the output stream.\n        '
        client = FakeAMP()
        clientTransport = StringTransport()
        client.makeConnection(clientTransport)
        client.callRemote(workercommands.Run, testCase='doesntexist')
        self.readStream = clientTransport.io
        self.readStream.seek(0, 0)
        main(self.fdopen)
        self.assertIn(b'StreamOpen', self.writeStream.getvalue())

    def test_readInterrupted(self):
        if False:
            while True:
                i = 10
        '\n        If reading the input stream fails with a C{IOError} with errno\n        C{EINTR}, L{main} ignores it and continues reading.\n        '
        excInfos = []

        class FakeStream:
            count = 0

            def read(oself, size):
                if False:
                    for i in range(10):
                        print('nop')
                oself.count += 1
                if oself.count == 1:
                    raise OSError(errno.EINTR)
                else:
                    excInfos.append(sys.exc_info())
                return b''
        self.readStream = FakeStream()
        main(self.fdopen)
        self.assertEqual(b'', self.writeStream.getvalue())
        self.assertEqual([(None, None, None)], excInfos)

    def test_otherReadError(self):
        if False:
            i = 10
            return i + 15
        '\n        L{main} only ignores C{IOError} with C{EINTR} errno: otherwise, the\n        error pops out.\n        '

        class FakeStream:
            count = 0

            def read(oself, size):
                if False:
                    return 10
                oself.count += 1
                if oself.count == 1:
                    raise OSError('Something else')
                return ''
        self.readStream = FakeStream()
        self.assertRaises(IOError, main, self.fdopen)