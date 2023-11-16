"""
Interfaces for Trial.

Maintainer: Jonathan Lange
"""
import zope.interface as zi

class ITestCase(zi.Interface):
    """
    The interface that a test case must implement in order to be used in Trial.
    """
    failureException = zi.Attribute('The exception class that is raised by failed assertions')

    def __call__(result):
        if False:
            i = 10
            return i + 15
        '\n        Run the test. Should always do exactly the same thing as run().\n        '

    def countTestCases():
        if False:
            while True:
                i = 10
        '\n        Return the number of tests in this test case. Usually 1.\n        '

    def id():
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a unique identifier for the test, usually the fully-qualified\n        Python name.\n        '

    def run(result):
        if False:
            print('Hello World!')
        '\n        Run the test, storing the results in C{result}.\n\n        @param result: A L{TestResult}.\n        '

    def shortDescription():
        if False:
            return 10
        '\n        Return a short description of the test.\n        '

class IReporter(zi.Interface):
    """
    I report results from a run of a test suite.
    """
    shouldStop = zi.Attribute('A boolean indicating that this reporter would like the test run to stop.')
    testsRun = zi.Attribute('\n        The number of tests that seem to have been run according to this\n        reporter.\n        ')

    def startTest(method):
        if False:
            print('Hello World!')
        '\n        Report the beginning of a run of a single test method.\n\n        @param method: an object that is adaptable to ITestMethod\n        '

    def stopTest(method):
        if False:
            while True:
                i = 10
        '\n        Report the status of a single test method\n\n        @param method: an object that is adaptable to ITestMethod\n        '

    def addSuccess(test):
        if False:
            return 10
        '\n        Record that test passed.\n        '

    def addError(test, error):
        if False:
            while True:
                i = 10
        '\n        Record that a test has raised an unexpected exception.\n\n        @param test: The test that has raised an error.\n        @param error: The error that the test raised. It will either be a\n            three-tuple in the style of C{sys.exc_info()} or a\n            L{Failure<twisted.python.failure.Failure>} object.\n        '

    def addFailure(test, failure):
        if False:
            print('Hello World!')
        '\n        Record that a test has failed with the given failure.\n\n        @param test: The test that has failed.\n        @param failure: The failure that the test failed with. It will\n            either be a three-tuple in the style of C{sys.exc_info()}\n            or a L{Failure<twisted.python.failure.Failure>} object.\n        '

    def addExpectedFailure(test, failure, todo=None):
        if False:
            while True:
                i = 10
        "\n        Record that the given test failed, and was expected to do so.\n\n        In Twisted 15.5 and prior, C{todo} was a mandatory parameter.\n\n        @type test: L{unittest.TestCase}\n        @param test: The test which this is about.\n        @type failure: L{failure.Failure}\n        @param failure: The error which this test failed with.\n        @type todo: L{unittest.Todo}\n        @param todo: The reason for the test's TODO status. If L{None}, a\n            generic reason is used.\n        "

    def addUnexpectedSuccess(test, todo=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Record that the given test failed, and was expected to do so.\n\n        In Twisted 15.5 and prior, C{todo} was a mandatory parameter.\n\n        @type test: L{unittest.TestCase}\n        @param test: The test which this is about.\n        @type todo: L{unittest.Todo}\n        @param todo: The reason for the test's TODO status. If L{None}, a\n            generic reason is used.\n        "

    def addSkip(test, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        Record that a test has been skipped for the given reason.\n\n        @param test: The test that has been skipped.\n        @param reason: An object that the test case has specified as the reason\n            for skipping the test.\n        '

    def wasSuccessful():
        if False:
            return 10
        '\n        Return a boolean indicating whether all test results that were reported\n        to this reporter were successful or not.\n        '

    def done():
        if False:
            while True:
                i = 10
        '\n        Called when the test run is complete.\n\n        This gives the result object an opportunity to display a summary of\n        information to the user. Once you have called C{done} on an\n        L{IReporter} object, you should assume that the L{IReporter} object is\n        no longer usable.\n        '