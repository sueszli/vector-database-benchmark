import sys
from robot.model import SuiteVisitor
from robot.result import TestCase, TestSuite
from robot.utils import plural_or_not as s, secs_to_timestr
from .highlighting import HighlightingStream
from ..loggerapi import LoggerApi

class DottedOutput(LoggerApi):

    def __init__(self, width=78, colors='AUTO', stdout=None, stderr=None):
        if False:
            while True:
                i = 10
        self.width = width
        self.stdout = HighlightingStream(stdout or sys.__stdout__, colors)
        self.stderr = HighlightingStream(stderr or sys.__stderr__, colors)
        self.markers_on_row = 0

    def start_suite(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        if not data.parent:
            count = data.test_count
            ts = ('test' if not data.rpa else 'task') + s(count)
            self.stdout.write(f"Running suite '{result.name}' with {count} {ts}.\n")
            self.stdout.write('=' * self.width + '\n')

    def end_test(self, data, result):
        if False:
            i = 10
            return i + 15
        if self.markers_on_row == self.width:
            self.stdout.write('\n')
            self.markers_on_row = 0
        self.markers_on_row += 1
        if result.passed:
            self.stdout.write('.')
        elif result.skipped:
            self.stdout.highlight('s', 'SKIP')
        elif result.tags.robot('exit'):
            self.stdout.write('x')
        else:
            self.stdout.highlight('F', 'FAIL')

    def end_suite(self, data, result):
        if False:
            i = 10
            return i + 15
        if not data.parent:
            self.stdout.write('\n')
            StatusReporter(self.stdout, self.width).report(result)
            self.stdout.write('\n')

    def message(self, msg):
        if False:
            return 10
        if msg.level in ('WARN', 'ERROR'):
            self.stderr.error(msg.message, msg.level)

    def output_file(self, name, path):
        if False:
            i = 10
            return i + 15
        self.stdout.write(f"{name + ':':8} {path}\n")

class StatusReporter(SuiteVisitor):

    def __init__(self, stream, width):
        if False:
            return 10
        self.stream = stream
        self.width = width

    def report(self, suite: TestSuite):
        if False:
            for i in range(10):
                print('nop')
        suite.visit(self)
        stats = suite.statistics
        ts = ('test' if not suite.rpa else 'task') + s(stats.total)
        elapsed = secs_to_timestr(suite.elapsed_time)
        self.stream.write(f"{'=' * self.width}\nRun suite '{suite.name}' with {stats.total} {ts} in {elapsed}.\n\n")
        ed = 'ED' if suite.status != 'SKIP' else 'PED'
        self.stream.highlight(suite.status + ed, suite.status)
        self.stream.write(f'\n{stats.message}\n')

    def visit_test(self, test: TestCase):
        if False:
            print('Hello World!')
        if test.failed and (not test.tags.robot('exit')):
            self.stream.write('-' * self.width + '\n')
            self.stream.highlight('FAIL')
            self.stream.write(f': {test.full_name}\n{test.message.strip()}\n')