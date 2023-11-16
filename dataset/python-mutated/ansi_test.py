import sys
from unittest import TestCase, main
from ..ansi import Back, Fore, Style
from ..ansitowin32 import AnsiToWin32
stdout_orig = sys.stdout
stderr_orig = sys.stderr

class AnsiTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(type(sys.stdout), AnsiToWin32)
        self.assertNotEqual(type(sys.stderr), AnsiToWin32)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig

    def testForeAttributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(Fore.BLACK, '\x1b[30m')
        self.assertEqual(Fore.RED, '\x1b[31m')
        self.assertEqual(Fore.GREEN, '\x1b[32m')
        self.assertEqual(Fore.YELLOW, '\x1b[33m')
        self.assertEqual(Fore.BLUE, '\x1b[34m')
        self.assertEqual(Fore.MAGENTA, '\x1b[35m')
        self.assertEqual(Fore.CYAN, '\x1b[36m')
        self.assertEqual(Fore.WHITE, '\x1b[37m')
        self.assertEqual(Fore.RESET, '\x1b[39m')
        self.assertEqual(Fore.LIGHTBLACK_EX, '\x1b[90m')
        self.assertEqual(Fore.LIGHTRED_EX, '\x1b[91m')
        self.assertEqual(Fore.LIGHTGREEN_EX, '\x1b[92m')
        self.assertEqual(Fore.LIGHTYELLOW_EX, '\x1b[93m')
        self.assertEqual(Fore.LIGHTBLUE_EX, '\x1b[94m')
        self.assertEqual(Fore.LIGHTMAGENTA_EX, '\x1b[95m')
        self.assertEqual(Fore.LIGHTCYAN_EX, '\x1b[96m')
        self.assertEqual(Fore.LIGHTWHITE_EX, '\x1b[97m')

    def testBackAttributes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(Back.BLACK, '\x1b[40m')
        self.assertEqual(Back.RED, '\x1b[41m')
        self.assertEqual(Back.GREEN, '\x1b[42m')
        self.assertEqual(Back.YELLOW, '\x1b[43m')
        self.assertEqual(Back.BLUE, '\x1b[44m')
        self.assertEqual(Back.MAGENTA, '\x1b[45m')
        self.assertEqual(Back.CYAN, '\x1b[46m')
        self.assertEqual(Back.WHITE, '\x1b[47m')
        self.assertEqual(Back.RESET, '\x1b[49m')
        self.assertEqual(Back.LIGHTBLACK_EX, '\x1b[100m')
        self.assertEqual(Back.LIGHTRED_EX, '\x1b[101m')
        self.assertEqual(Back.LIGHTGREEN_EX, '\x1b[102m')
        self.assertEqual(Back.LIGHTYELLOW_EX, '\x1b[103m')
        self.assertEqual(Back.LIGHTBLUE_EX, '\x1b[104m')
        self.assertEqual(Back.LIGHTMAGENTA_EX, '\x1b[105m')
        self.assertEqual(Back.LIGHTCYAN_EX, '\x1b[106m')
        self.assertEqual(Back.LIGHTWHITE_EX, '\x1b[107m')

    def testStyleAttributes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(Style.DIM, '\x1b[2m')
        self.assertEqual(Style.NORMAL, '\x1b[22m')
        self.assertEqual(Style.BRIGHT, '\x1b[1m')
if __name__ == '__main__':
    main()