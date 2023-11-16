"""Pre-run modifier that selects only every Xth test for execution.

Starts from the first test by default. Tests are selected per suite.
"""
from robot.api import SuiteVisitor

class SelectEveryXthTest(SuiteVisitor):

    def __init__(self, x: int, start: int=0):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.start = start

    def start_suite(self, suite):
        if False:
            print('Hello World!')
        "Modify suite's tests to contain only every Xth."
        suite.tests = suite.tests[self.start::self.x]

    def end_suite(self, suite):
        if False:
            i = 10
            return i + 15
        'Remove suites that are empty after removing tests.'
        suite.suites = [s for s in suite.suites if s.test_count > 0]

    def visit_test(self, test):
        if False:
            print('Hello World!')
        'Avoid visiting tests and their keywords to save a little time.'
        pass