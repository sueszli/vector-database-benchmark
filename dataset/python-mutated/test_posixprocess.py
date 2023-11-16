"""
Tests for POSIX-based L{IReactorProcess} implementations.
"""
import errno
import os
import sys
from typing import Optional
platformSkip: Optional[str]
try:
    import fcntl
except ImportError:
    platformSkip = 'non-POSIX platform'
else:
    from twisted.internet import process
    platformSkip = None
from twisted.trial.unittest import TestCase

class FakeFile:
    """
    A dummy file object which records when it is closed.
    """

    def __init__(self, testcase, fd):
        if False:
            while True:
                i = 10
        self.testcase = testcase
        self.fd = fd

    def close(self):
        if False:
            print('Hello World!')
        self.testcase._files.remove(self.fd)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        self.close()

class FakeResourceModule:
    """
    Fake version of L{resource} which hard-codes a particular rlimit for maximum
    open files.

    @ivar _limit: The value to return for the hard limit of number of open files.
    """
    RLIMIT_NOFILE = 1

    def __init__(self, limit):
        if False:
            print('Hello World!')
        self._limit = limit

    def getrlimit(self, no):
        if False:
            print('Hello World!')
        '\n        A fake of L{resource.getrlimit} which returns a pre-determined result.\n        '
        if no == self.RLIMIT_NOFILE:
            return [0, self._limit]
        return [123, 456]

class FDDetectorTests(TestCase):
    """
    Tests for _FDDetector class in twisted.internet.process, which detects
    which function to drop in place for the _listOpenFDs method.

    @ivar devfs: A flag indicating whether the filesystem fake will indicate
        that /dev/fd exists.

    @ivar accurateDevFDResults: A flag indicating whether the /dev/fd fake
        returns accurate open file information.

    @ivar procfs: A flag indicating whether the filesystem fake will indicate
        that /proc/<pid>/fd exists.
    """
    skip = platformSkip
    devfs = False
    accurateDevFDResults = False
    procfs = False

    def getpid(self):
        if False:
            print('Hello World!')
        '\n        Fake os.getpid, always return the same thing\n        '
        return 123

    def listdir(self, arg):
        if False:
            i = 10
            return i + 15
        "\n        Fake os.listdir, depending on what mode we're in to simulate behaviour.\n\n        @param arg: the directory to list\n        "
        accurate = map(str, self._files)
        if self.procfs and arg == '/proc/%d/fd' % (self.getpid(),):
            return accurate
        if self.devfs and arg == '/dev/fd':
            if self.accurateDevFDResults:
                return accurate
            return ['0', '1', '2']
        raise OSError()

    def openfile(self, fname, mode):
        if False:
            print('Hello World!')
        '\n        This is a mock for L{open}.  It keeps track of opened files so extra\n        descriptors can be returned from the mock for L{os.listdir} when used on\n        one of the list-of-filedescriptors directories.\n\n        A L{FakeFile} is returned which can be closed to remove the new\n        descriptor from the open list.\n        '
        f = FakeFile(self, min(set(range(1024)) - set(self._files)))
        self._files.append(f.fd)
        return f

    def hideResourceModule(self):
        if False:
            return 10
        '\n        Make the L{resource} module unimportable for the remainder of the\n        current test method.\n        '
        sys.modules['resource'] = None

    def revealResourceModule(self, limit):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make a L{FakeResourceModule} instance importable at the L{resource}\n        name.\n\n        @param limit: The value which will be returned for the hard limit of\n            number of open files by the fake resource module's C{getrlimit}\n            function.\n        "
        sys.modules['resource'] = FakeResourceModule(limit)

    def replaceResourceModule(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Restore the original resource module to L{sys.modules}.\n        '
        if value is None:
            try:
                del sys.modules['resource']
            except KeyError:
                pass
        else:
            sys.modules['resource'] = value

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Set up the tests, giving ourselves a detector object to play with and\n        setting up its testable knobs to refer to our mocked versions.\n        '
        self.detector = process._FDDetector()
        self.detector.listdir = self.listdir
        self.detector.getpid = self.getpid
        self.detector.openfile = self.openfile
        self._files = [0, 1, 2]
        self.addCleanup(self.replaceResourceModule, sys.modules.get('resource'))

    def test_selectFirstWorking(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{FDDetector._getImplementation} returns the first method from its\n        C{_implementations} list which returns results which reflect a newly\n        opened file descriptor.\n        '

        def failWithException():
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError('This does not work')

        def failWithWrongResults():
            if False:
                return 10
            return [0, 1, 2]

        def correct():
            if False:
                i = 10
                return i + 15
            return self._files[:]
        self.detector._implementations = [failWithException, failWithWrongResults, correct]
        self.assertIs(correct, self.detector._getImplementation())

    def test_selectLast(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{FDDetector._getImplementation} returns the last method from its\n        C{_implementations} list if none of the implementations manage to return\n        results which reflect a newly opened file descriptor.\n        '

        def failWithWrongResults():
            if False:
                i = 10
                return i + 15
            return [3, 5, 9]

        def failWithOtherWrongResults():
            if False:
                for i in range(10):
                    print('nop')
            return [0, 1, 2]
        self.detector._implementations = [failWithWrongResults, failWithOtherWrongResults]
        self.assertIs(failWithOtherWrongResults, self.detector._getImplementation())

    def test_identityOfListOpenFDsChanges(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check that the identity of _listOpenFDs changes after running\n        _listOpenFDs the first time, but not after the second time it's run.\n\n        In other words, check that the monkey patching actually works.\n        "
        detector = process._FDDetector()
        first = detector._listOpenFDs.__name__
        detector._listOpenFDs()
        second = detector._listOpenFDs.__name__
        detector._listOpenFDs()
        third = detector._listOpenFDs.__name__
        self.assertNotEqual(first, second)
        self.assertEqual(second, third)

    def test_devFDImplementation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_FDDetector._devFDImplementation} raises L{OSError} if there is no\n        I{/dev/fd} directory, otherwise it returns the basenames of its children\n        interpreted as integers.\n        '
        self.devfs = False
        self.assertRaises(OSError, self.detector._devFDImplementation)
        self.devfs = True
        self.accurateDevFDResults = False
        self.assertEqual([0, 1, 2], self.detector._devFDImplementation())

    def test_procFDImplementation(self):
        if False:
            while True:
                i = 10
        '\n        L{_FDDetector._procFDImplementation} raises L{OSError} if there is no\n        I{/proc/<pid>/fd} directory, otherwise it returns the basenames of its\n        children interpreted as integers.\n        '
        self.procfs = False
        self.assertRaises(OSError, self.detector._procFDImplementation)
        self.procfs = True
        self.assertEqual([0, 1, 2], self.detector._procFDImplementation())

    def test_resourceFDImplementation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_FDDetector._fallbackFDImplementation} uses the L{resource} module if\n        it is available, returning a range of integers from 0 to the\n        minimum of C{1024} and the hard I{NOFILE} limit.\n        '
        self.revealResourceModule(512)
        self.assertEqual(list(range(512)), list(self.detector._fallbackFDImplementation()))
        self.revealResourceModule(2048)
        self.assertEqual(list(range(1024)), list(self.detector._fallbackFDImplementation()))

    def test_fallbackFDImplementation(self):
        if False:
            return 10
        '\n        L{_FDDetector._fallbackFDImplementation}, the implementation of last\n        resort, succeeds with a fixed range of integers from 0 to 1024 when the\n        L{resource} module is not importable.\n        '
        self.hideResourceModule()
        self.assertEqual(list(range(1024)), list(self.detector._fallbackFDImplementation()))

class FileDescriptorTests(TestCase):
    """
    Tests for L{twisted.internet.process._listOpenFDs}
    """
    skip = platformSkip

    def test_openFDs(self):
        if False:
            i = 10
            return i + 15
        '\n        File descriptors returned by L{_listOpenFDs} are mostly open.\n\n        This test assumes that zero-legth writes fail with EBADF on closed\n        file descriptors.\n        '
        for fd in process._listOpenFDs():
            try:
                fcntl.fcntl(fd, fcntl.F_GETFL)
            except OSError as err:
                self.assertEqual(errno.EBADF, err.errno, 'fcntl(%d, F_GETFL) failed with unexpected errno %d' % (fd, err.errno))

    def test_expectedFDs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_listOpenFDs} lists expected file descriptors.\n        '
        f = open(os.devnull)
        openfds = process._listOpenFDs()
        self.assertIn(f.fileno(), openfds)
        fd = os.dup(f.fileno())
        self.assertTrue(fd > f.fileno(), 'Expected duplicate file descriptor to be greater than original')
        try:
            f.close()
            self.assertIn(fd, process._listOpenFDs())
        finally:
            os.close(fd)
        self.assertNotIn(fd, process._listOpenFDs())