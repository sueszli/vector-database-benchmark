import builtins
import os
import select
import socket
import unittest
import errno
from errno import EEXIST

class SubOSError(OSError):
    pass

class SubOSErrorWithInit(OSError):

    def __init__(self, message, bar):
        if False:
            i = 10
            return i + 15
        self.bar = bar
        super().__init__(message)

class SubOSErrorWithNew(OSError):

    def __new__(cls, message, baz):
        if False:
            while True:
                i = 10
        self = super().__new__(cls, message)
        self.baz = baz
        return self

class SubOSErrorCombinedInitFirst(SubOSErrorWithInit, SubOSErrorWithNew):
    pass

class SubOSErrorCombinedNewFirst(SubOSErrorWithNew, SubOSErrorWithInit):
    pass

class SubOSErrorWithStandaloneInit(OSError):

    def __init__(self):
        if False:
            return 10
        pass

class HierarchyTest(unittest.TestCase):

    def test_builtin_errors(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(OSError.__name__, 'OSError')
        self.assertIs(IOError, OSError)
        self.assertIs(EnvironmentError, OSError)

    def test_socket_errors(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(socket.error, OSError)
        self.assertIs(socket.gaierror.__base__, OSError)
        self.assertIs(socket.herror.__base__, OSError)
        self.assertIs(socket.timeout, TimeoutError)

    def test_select_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(select.error, OSError)
    _pep_map = '\n        +-- BlockingIOError        EAGAIN, EALREADY, EWOULDBLOCK, EINPROGRESS\n        +-- ChildProcessError                                          ECHILD\n        +-- ConnectionError\n            +-- BrokenPipeError                              EPIPE, ESHUTDOWN\n            +-- ConnectionAbortedError                           ECONNABORTED\n            +-- ConnectionRefusedError                           ECONNREFUSED\n            +-- ConnectionResetError                               ECONNRESET\n        +-- FileExistsError                                            EEXIST\n        +-- FileNotFoundError                                          ENOENT\n        +-- InterruptedError                                            EINTR\n        +-- IsADirectoryError                                          EISDIR\n        +-- NotADirectoryError                                        ENOTDIR\n        +-- PermissionError                                     EACCES, EPERM\n        +-- ProcessLookupError                                          ESRCH\n        +-- TimeoutError                                            ETIMEDOUT\n    '

    def _make_map(s):
        if False:
            print('Hello World!')
        _map = {}
        for line in s.splitlines():
            line = line.strip('+- ')
            if not line:
                continue
            (excname, _, errnames) = line.partition(' ')
            for errname in filter(None, errnames.strip().split(', ')):
                _map[getattr(errno, errname)] = getattr(builtins, excname)
        return _map
    _map = _make_map(_pep_map)

    def test_errno_mapping(self):
        if False:
            i = 10
            return i + 15
        e = OSError(EEXIST, 'Bad file descriptor')
        self.assertIs(type(e), FileExistsError)
        for (errcode, exc) in self._map.items():
            e = OSError(errcode, 'Some message')
            self.assertIs(type(e), exc)
        othercodes = set(errno.errorcode) - set(self._map)
        for errcode in othercodes:
            e = OSError(errcode, 'Some message')
            self.assertIs(type(e), OSError)

    def test_try_except(self):
        if False:
            for i in range(10):
                print('nop')
        filename = 'some_hopefully_non_existing_file'
        try:
            open(filename)
        except FileNotFoundError:
            pass
        else:
            self.fail('should have raised a FileNotFoundError')
        self.assertFalse(os.path.exists(filename))
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
        else:
            self.fail('should have raised a FileNotFoundError')

class AttributesTest(unittest.TestCase):

    def test_windows_error(self):
        if False:
            return 10
        if os.name == 'nt':
            self.assertIn('winerror', dir(OSError))
        else:
            self.assertNotIn('winerror', dir(OSError))

    def test_posix_error(self):
        if False:
            for i in range(10):
                print('nop')
        e = OSError(EEXIST, 'File already exists', 'foo.txt')
        self.assertEqual(e.errno, EEXIST)
        self.assertEqual(e.args[0], EEXIST)
        self.assertEqual(e.strerror, 'File already exists')
        self.assertEqual(e.filename, 'foo.txt')
        if os.name == 'nt':
            self.assertEqual(e.winerror, None)

    @unittest.skipUnless(os.name == 'nt', 'Windows-specific test')
    def test_errno_translation(self):
        if False:
            print('Hello World!')
        e = OSError(0, 'File already exists', 'foo.txt', 183)
        self.assertEqual(e.winerror, 183)
        self.assertEqual(e.errno, EEXIST)
        self.assertEqual(e.args[0], EEXIST)
        self.assertEqual(e.strerror, 'File already exists')
        self.assertEqual(e.filename, 'foo.txt')

    def test_blockingioerror(self):
        if False:
            while True:
                i = 10
        args = ('a', 'b', 'c', 'd', 'e')
        for n in range(6):
            e = BlockingIOError(*args[:n])
            with self.assertRaises(AttributeError):
                e.characters_written
            with self.assertRaises(AttributeError):
                del e.characters_written
        e = BlockingIOError('a', 'b', 3)
        self.assertEqual(e.characters_written, 3)
        e.characters_written = 5
        self.assertEqual(e.characters_written, 5)
        del e.characters_written
        with self.assertRaises(AttributeError):
            e.characters_written

class ExplicitSubclassingTest(unittest.TestCase):

    def test_errno_mapping(self):
        if False:
            for i in range(10):
                print('nop')
        e = SubOSError(EEXIST, 'Bad file descriptor')
        self.assertIs(type(e), SubOSError)

    def test_init_overridden(self):
        if False:
            print('Hello World!')
        e = SubOSErrorWithInit('some message', 'baz')
        self.assertEqual(e.bar, 'baz')
        self.assertEqual(e.args, ('some message',))

    def test_init_kwdargs(self):
        if False:
            print('Hello World!')
        e = SubOSErrorWithInit('some message', bar='baz')
        self.assertEqual(e.bar, 'baz')
        self.assertEqual(e.args, ('some message',))

    def test_new_overridden(self):
        if False:
            for i in range(10):
                print('nop')
        e = SubOSErrorWithNew('some message', 'baz')
        self.assertEqual(e.baz, 'baz')
        self.assertEqual(e.args, ('some message',))

    def test_new_kwdargs(self):
        if False:
            while True:
                i = 10
        e = SubOSErrorWithNew('some message', baz='baz')
        self.assertEqual(e.baz, 'baz')
        self.assertEqual(e.args, ('some message',))

    def test_init_new_overridden(self):
        if False:
            print('Hello World!')
        e = SubOSErrorCombinedInitFirst('some message', 'baz')
        self.assertEqual(e.bar, 'baz')
        self.assertEqual(e.baz, 'baz')
        self.assertEqual(e.args, ('some message',))
        e = SubOSErrorCombinedNewFirst('some message', 'baz')
        self.assertEqual(e.bar, 'baz')
        self.assertEqual(e.baz, 'baz')
        self.assertEqual(e.args, ('some message',))

    def test_init_standalone(self):
        if False:
            return 10
        e = SubOSErrorWithStandaloneInit()
        self.assertEqual(e.args, ())
        self.assertEqual(str(e), '')
if __name__ == '__main__':
    unittest.main()