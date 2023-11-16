"""
The reporter is not made to support concurrent test running, so we will
hold test results in here and only send them to the reporter once the
test is over.

@since: 12.3
"""
from types import TracebackType
from typing import Optional, Tuple, Union
from zope.interface import implementer
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from ..itrial import IReporter, ITestCase
ReporterFailure = Union[Failure, Tuple[type, Exception, TracebackType]]

@implementer(IReporter)
class DistReporter(proxyForInterface(IReporter)):
    """
    See module docstring.
    """

    def __init__(self, original):
        if False:
            i = 10
            return i + 15
        super().__init__(original)
        self.running = {}

    def startTest(self, test):
        if False:
            i = 10
            return i + 15
        '\n        Queue test starting.\n        '
        self.running[test.id()] = []
        self.running[test.id()].append((self.original.startTest, test))

    def addFailure(self, test: ITestCase, fail: ReporterFailure) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Queue adding a failure.\n        '
        self.running[test.id()].append((self.original.addFailure, test, fail))

    def addError(self, test: ITestCase, error: ReporterFailure) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Queue error adding.\n        '
        self.running[test.id()].append((self.original.addError, test, error))

    def addSkip(self, test, reason):
        if False:
            while True:
                i = 10
        '\n        Queue adding a skip.\n        '
        self.running[test.id()].append((self.original.addSkip, test, reason))

    def addUnexpectedSuccess(self, test, todo=None):
        if False:
            while True:
                i = 10
        '\n        Queue adding an unexpected success.\n        '
        self.running[test.id()].append((self.original.addUnexpectedSuccess, test, todo))

    def addExpectedFailure(self, test: ITestCase, error: ReporterFailure, todo: Optional[str]=None) -> None:
        if False:
            return 10
        '\n        Queue adding an expected failure.\n        '
        self.running[test.id()].append((self.original.addExpectedFailure, test, error, todo))

    def addSuccess(self, test):
        if False:
            print('Hello World!')
        '\n        Queue adding a success.\n        '
        self.running[test.id()].append((self.original.addSuccess, test))

    def stopTest(self, test):
        if False:
            while True:
                i = 10
        '\n        Queue stopping the test, then unroll the queue.\n        '
        self.running[test.id()].append((self.original.stopTest, test))
        for step in self.running[test.id()]:
            step[0](*step[1:])
        del self.running[test.id()]