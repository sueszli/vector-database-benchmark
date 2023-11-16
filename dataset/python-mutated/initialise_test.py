import sys
from unittest import TestCase, main, skipUnless
try:
    from unittest.mock import patch, Mock
except ImportError:
    from mock import patch, Mock
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
orig_stdout = sys.stdout
orig_stderr = sys.stderr

class InitTest(TestCase):

    @skipUnless(sys.stdout.isatty(), 'sys.stdout is not a tty')
    def setUp(self):
        if False:
            while True:
                i = 10
        self.assertNotWrapped()

    def tearDown(self):
        if False:
            print('Hello World!')
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def assertWrapped(self):
        if False:
            while True:
                i = 10
        self.assertIsNot(sys.stdout, orig_stdout, 'stdout should be wrapped')
        self.assertIsNot(sys.stderr, orig_stderr, 'stderr should be wrapped')
        self.assertTrue(isinstance(sys.stdout, StreamWrapper), 'bad stdout wrapper')
        self.assertTrue(isinstance(sys.stderr, StreamWrapper), 'bad stderr wrapper')

    def assertNotWrapped(self):
        if False:
            while True:
                i = 10
        self.assertIs(sys.stdout, orig_stdout, 'stdout should not be wrapped')
        self.assertIs(sys.stderr, orig_stderr, 'stderr should not be wrapped')

    @patch('colorama.initialise.reset_all')
    @patch('colorama.ansitowin32.winapi_test', lambda *_: True)
    @patch('colorama.ansitowin32.enable_vt_processing', lambda *_: False)
    def testInitWrapsOnWindows(self, _):
        if False:
            return 10
        with osname('nt'):
            init()
            self.assertWrapped()

    @patch('colorama.initialise.reset_all')
    @patch('colorama.ansitowin32.winapi_test', lambda *_: False)
    def testInitDoesntWrapOnEmulatedWindows(self, _):
        if False:
            for i in range(10):
                print('nop')
        with osname('nt'):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapOnNonWindows(self):
        if False:
            while True:
                i = 10
        with osname('posix'):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapIfNone(self):
        if False:
            for i in range(10):
                print('nop')
        with replace_by(None):
            init()
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)

    def testInitAutoresetOnWrapsOnAllPlatforms(self):
        if False:
            print('Hello World!')
        with osname('posix'):
            init(autoreset=True)
            self.assertWrapped()

    def testInitWrapOffDoesntWrapOnWindows(self):
        if False:
            i = 10
            return i + 15
        with osname('nt'):
            init(wrap=False)
            self.assertNotWrapped()

    def testInitWrapOffIncompatibleWithAutoresetOn(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, lambda : init(autoreset=True, wrap=False))

    @patch('colorama.win32.SetConsoleTextAttribute')
    @patch('colorama.initialise.AnsiToWin32')
    def testAutoResetPassedOn(self, mockATW32, _):
        if False:
            for i in range(10):
                print('nop')
        with osname('nt'):
            init(autoreset=True)
            self.assertEqual(len(mockATW32.call_args_list), 2)
            self.assertEqual(mockATW32.call_args_list[1][1]['autoreset'], True)
            self.assertEqual(mockATW32.call_args_list[0][1]['autoreset'], True)

    @patch('colorama.initialise.AnsiToWin32')
    def testAutoResetChangeable(self, mockATW32):
        if False:
            for i in range(10):
                print('nop')
        with osname('nt'):
            init()
            init(autoreset=True)
            self.assertEqual(len(mockATW32.call_args_list), 4)
            self.assertEqual(mockATW32.call_args_list[2][1]['autoreset'], True)
            self.assertEqual(mockATW32.call_args_list[3][1]['autoreset'], True)
            init()
            self.assertEqual(len(mockATW32.call_args_list), 6)
            self.assertEqual(mockATW32.call_args_list[4][1]['autoreset'], False)
            self.assertEqual(mockATW32.call_args_list[5][1]['autoreset'], False)

    @patch('colorama.initialise.atexit.register')
    def testAtexitRegisteredOnlyOnce(self, mockRegister):
        if False:
            while True:
                i = 10
        init()
        self.assertTrue(mockRegister.called)
        mockRegister.reset_mock()
        init()
        self.assertFalse(mockRegister.called)

class JustFixWindowsConsoleTest(TestCase):

    def _reset(self):
        if False:
            while True:
                i = 10
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self._reset()

    @patch('colorama.ansitowin32.winapi_test', lambda : True)
    def testJustFixWindowsConsole(self):
        if False:
            while True:
                i = 10
        if sys.platform != 'win32':
            just_fix_windows_console()
            self.assertIs(sys.stdout, orig_stdout)
            self.assertIs(sys.stderr, orig_stderr)
        else:

            def fake_std():
                if False:
                    while True:
                        i = 10
                stdout = Mock()
                stdout.closed = False
                stdout.isatty.return_value = False
                stdout.fileno.return_value = 1
                sys.stdout = stdout
                stderr = Mock()
                stderr.closed = False
                stderr.isatty.return_value = True
                stderr.fileno.return_value = 2
                sys.stderr = stderr
            for native_ansi in [False, True]:
                with patch('colorama.ansitowin32.enable_vt_processing', lambda *_: native_ansi):
                    self._reset()
                    fake_std()
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(sys.stdout, prev_stdout)
                    if native_ansi:
                        self.assertIs(sys.stderr, prev_stderr)
                    else:
                        self.assertIsNot(sys.stderr, prev_stderr)
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(sys.stdout, prev_stdout)
                    self.assertIs(sys.stderr, prev_stderr)
                    self._reset()
                    fake_std()
                    init()
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(prev_stdout, sys.stdout)
                    self.assertIs(prev_stderr, sys.stderr)
if __name__ == '__main__':
    main()