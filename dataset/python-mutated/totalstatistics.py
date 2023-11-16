from collections.abc import Iterator
from robot.utils import test_or_task
from .stats import TotalStat
from .visitor import SuiteVisitor

class TotalStatistics:
    """Container for total statistics."""

    def __init__(self, rpa: bool=False):
        if False:
            return 10
        self._stat = TotalStat(test_or_task('All {Test}s', rpa))
        self._rpa = rpa

    def visit(self, visitor):
        if False:
            print('Hello World!')
        visitor.visit_total_statistics(self._stat)

    def __iter__(self) -> 'Iterator[TotalStat]':
        if False:
            while True:
                i = 10
        yield self._stat

    @property
    def total(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._stat.total

    @property
    def passed(self) -> int:
        if False:
            return 10
        return self._stat.passed

    @property
    def skipped(self) -> int:
        if False:
            return 10
        return self._stat.skipped

    @property
    def failed(self) -> int:
        if False:
            print('Hello World!')
        return self._stat.failed

    def add_test(self, test):
        if False:
            print('Hello World!')
        self._stat.add_test(test)

    @property
    def message(self) -> str:
        if False:
            while True:
                i = 10
        'String representation of the statistics.\n\n        For example::\n            2 tests, 1 passed, 1 failed\n        '
        test_or_task = 'test' if not self._rpa else 'task'
        (total, end, passed, failed, skipped) = self._get_counts()
        template = '%d %s%s, %d passed, %d failed'
        if skipped:
            return (template + ', %d skipped') % (total, test_or_task, end, passed, failed, skipped)
        return template % (total, test_or_task, end, passed, failed)

    def _get_counts(self):
        if False:
            for i in range(10):
                print('nop')
        ending = 's' if self.total != 1 else ''
        return (self.total, ending, self.passed, self.failed, self.skipped)

class TotalStatisticsBuilder(SuiteVisitor):

    def __init__(self, suite=None, rpa=False):
        if False:
            return 10
        self.stats = TotalStatistics(rpa)
        if suite:
            suite.visit(self)

    def add_test(self, test):
        if False:
            i = 10
            return i + 15
        self.stats.add_test(test)

    def visit_test(self, test):
        if False:
            i = 10
            return i + 15
        self.add_test(test)

    def visit_keyword(self, kw):
        if False:
            while True:
                i = 10
        pass