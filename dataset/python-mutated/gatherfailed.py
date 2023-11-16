from robot.errors import DataError
from robot.model import SuiteVisitor
from robot.result import ExecutionResult
from robot.utils import get_error_message, glob_escape

class GatherFailedTests(SuiteVisitor):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.tests = []

    def visit_test(self, test):
        if False:
            while True:
                i = 10
        if test.failed:
            self.tests.append(glob_escape(test.full_name))

    def visit_keyword(self, kw):
        if False:
            print('Hello World!')
        pass

class GatherFailedSuites(SuiteVisitor):

    def __init__(self):
        if False:
            print('Hello World!')
        self.suites = []

    def start_suite(self, suite):
        if False:
            while True:
                i = 10
        if any((test.failed for test in suite.tests)):
            self.suites.append(glob_escape(suite.full_name))

    def visit_test(self, test):
        if False:
            return 10
        pass

    def visit_keyword(self, kw):
        if False:
            i = 10
            return i + 15
        pass

def gather_failed_tests(output, empty_suite_ok=False):
    if False:
        print('Hello World!')
    if output is None:
        return None
    gatherer = GatherFailedTests()
    tests_or_tasks = 'tests or tasks'
    try:
        suite = ExecutionResult(output, include_keywords=False).suite
        suite.visit(gatherer)
        tests_or_tasks = 'tests' if not suite.rpa else 'tasks'
        if not gatherer.tests and (not empty_suite_ok):
            raise DataError('All %s passed.' % tests_or_tasks)
    except Exception:
        raise DataError("Collecting failed %s from '%s' failed: %s" % (tests_or_tasks, output, get_error_message()))
    return gatherer.tests

def gather_failed_suites(output, empty_suite_ok=False):
    if False:
        for i in range(10):
            print('nop')
    if output is None:
        return None
    gatherer = GatherFailedSuites()
    try:
        ExecutionResult(output, include_keywords=False).suite.visit(gatherer)
        if not gatherer.suites and (not empty_suite_ok):
            raise DataError('All suites passed.')
    except Exception:
        raise DataError("Collecting failed suites from '%s' failed: %s" % (output, get_error_message()))
    return gatherer.suites