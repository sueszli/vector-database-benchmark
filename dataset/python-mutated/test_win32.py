"""
Tests for L{twisted.python.win32}.
"""
from twisted.python import reflect, win32
from twisted.trial import unittest

class CommandLineQuotingTests(unittest.TestCase):
    """
    Tests for L{cmdLineQuote}.
    """

    def test_argWithoutSpaces(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling C{cmdLineQuote} with an argument with no spaces returns\n        the argument unchanged.\n        '
        self.assertEqual(win32.cmdLineQuote('an_argument'), 'an_argument')

    def test_argWithSpaces(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Calling C{cmdLineQuote} with an argument containing spaces returns\n        the argument surrounded by quotes.\n        '
        self.assertEqual(win32.cmdLineQuote('An Argument'), '"An Argument"')

    def test_emptyStringArg(self) -> None:
        if False:
            print('Hello World!')
        '\n        Calling C{cmdLineQuote} with an empty string returns a quoted empty\n        string.\n        '
        self.assertEqual(win32.cmdLineQuote(''), '""')

class DeprecationTests(unittest.TestCase):
    """
    Tests for deprecated (Fake)WindowsError.
    """

    def test_deprecation_FakeWindowsError(self) -> None:
        if False:
            print('Hello World!')
        'Importing C{FakeWindowsError} triggers a L{DeprecationWarning}.'
        self.assertWarns(DeprecationWarning, "twisted.python.win32.FakeWindowsError was deprecated in Twisted 21.2.0: Catch OSError and check presence of 'winerror' attribute.", reflect.__file__, lambda : reflect.namedAny('twisted.python.win32.FakeWindowsError'))

    def test_deprecation_WindowsError(self) -> None:
        if False:
            while True:
                i = 10
        'Importing C{WindowsError} triggers a L{DeprecationWarning}.'
        self.assertWarns(DeprecationWarning, "twisted.python.win32.WindowsError was deprecated in Twisted 21.2.0: Catch OSError and check presence of 'winerror' attribute.", reflect.__file__, lambda : reflect.namedAny('twisted.python.win32.WindowsError'))