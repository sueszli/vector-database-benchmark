"""
Tests for L{twisted.application.runner._exit}.
"""
from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit

class ExitTests(twisted.trial.unittest.TestCase):
    """
    Tests for L{exit}.
    """

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.exit = DummyExit()
        self.patch(_exit, 'sysexit', self.exit)

    def test_exitStatusInt(self) -> None:
        if False:
            return 10
        '\n        L{exit} given an L{int} status code will pass it to L{sys.exit}.\n        '
        status = 1234
        exit(status)
        self.assertEqual(self.exit.arg, status)

    def test_exitConstant(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{exit} given a L{ValueConstant} status code passes the corresponding\n        value to L{sys.exit}.\n        '
        status = ExitStatus.EX_CONFIG
        exit(status)
        self.assertEqual(self.exit.arg, status.value)

    def test_exitMessageZero(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{exit} given a status code of zero (C{0}) writes the given message to\n        standard output.\n        '
        out = StringIO()
        self.patch(_exit, 'stdout', out)
        message = 'Hello, world.'
        exit(0, message)
        self.assertEqual(out.getvalue(), message + '\n')

    def test_exitMessageNonZero(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{exit} given a non-zero status code writes the given message to\n        standard error.\n        '
        out = StringIO()
        self.patch(_exit, 'stderr', out)
        message = 'Hello, world.'
        exit(64, message)
        self.assertEqual(out.getvalue(), message + '\n')

class DummyExit:
    """
    Stub for L{sys.exit} that remembers whether it's been called and, if it
    has, what argument it was given.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.exited = False

    def __call__(self, arg: Optional[Union[int, str]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert not self.exited
        self.arg = arg
        self.exited = True