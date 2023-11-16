from .stats import SuiteStat

class SuiteStatistics:
    """Container for suite statistics."""

    def __init__(self, suite):
        if False:
            print('Hello World!')
        self.stat = SuiteStat(suite)
        self.suites = []

    def visit(self, visitor):
        if False:
            i = 10
            return i + 15
        visitor.visit_suite_statistics(self)

    def __iter__(self):
        if False:
            return 10
        yield self.stat
        for child in self.suites:
            for stat in child:
                yield stat

class SuiteStatisticsBuilder:

    def __init__(self, suite_stat_level):
        if False:
            return 10
        self._suite_stat_level = suite_stat_level
        self._stats_stack = []
        self.stats = None

    @property
    def current(self):
        if False:
            while True:
                i = 10
        return self._stats_stack[-1] if self._stats_stack else None

    def start_suite(self, suite):
        if False:
            print('Hello World!')
        self._stats_stack.append(SuiteStatistics(suite))
        if self.stats is None:
            self.stats = self.current

    def add_test(self, test):
        if False:
            i = 10
            return i + 15
        self.current.stat.add_test(test)

    def end_suite(self):
        if False:
            print('Hello World!')
        stats = self._stats_stack.pop()
        if self.current:
            self.current.stat.add_stat(stats.stat)
            if self._is_child_included():
                self.current.suites.append(stats)

    def _is_child_included(self):
        if False:
            i = 10
            return i + 15
        return self._include_all_levels() or self._below_threshold()

    def _include_all_levels(self):
        if False:
            return 10
        return self._suite_stat_level == -1

    def _below_threshold(self):
        if False:
            i = 10
            return i + 15
        return len(self._stats_stack) < self._suite_stat_level