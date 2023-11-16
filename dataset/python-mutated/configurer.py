from robot.utils import seq2str
from robot.errors import DataError
from .visitor import SuiteVisitor

class SuiteConfigurer(SuiteVisitor):

    def __init__(self, name=None, doc=None, metadata=None, set_tags=None, include_tags=None, exclude_tags=None, include_suites=None, include_tests=None, empty_suite_ok=False):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.doc = doc
        self.metadata = metadata
        self.set_tags = set_tags or []
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags
        self.include_suites = include_suites
        self.include_tests = include_tests
        self.empty_suite_ok = empty_suite_ok

    @property
    def add_tags(self):
        if False:
            i = 10
            return i + 15
        return [t for t in self.set_tags if not t.startswith('-')]

    @property
    def remove_tags(self):
        if False:
            return 10
        return [t[1:] for t in self.set_tags if t.startswith('-')]

    def visit_suite(self, suite):
        if False:
            while True:
                i = 10
        self._set_suite_attributes(suite)
        self._filter(suite)
        suite.set_tags(self.add_tags, self.remove_tags)

    def _set_suite_attributes(self, suite):
        if False:
            return 10
        if self.name:
            suite.name = self.name
        if self.doc:
            suite.doc = self.doc
        if self.metadata:
            suite.metadata.update(self.metadata)

    def _filter(self, suite):
        if False:
            while True:
                i = 10
        name = suite.name
        suite.filter(self.include_suites, self.include_tests, self.include_tags, self.exclude_tags)
        if not (suite.has_tests or self.empty_suite_ok):
            self._raise_no_tests_or_tasks_error(name, suite.rpa)

    def _raise_no_tests_or_tasks_error(self, name, rpa):
        if False:
            for i in range(10):
                print('nop')
        parts = [{False: 'tests', True: 'tasks', None: 'tests or tasks'}[rpa], self._get_test_selector_msgs(), self._get_suite_selector_msg()]
        raise DataError(f"Suite '{name}' contains no {' '.join((p for p in parts if p))}.")

    def _get_test_selector_msgs(self):
        if False:
            return 10
        parts = []
        for (separator, explanation, selectors) in [(None, 'matching name', self.include_tests), ('or', 'matching tags', self.include_tags), ('and', 'not matching tags', self.exclude_tags)]:
            if selectors:
                if parts:
                    parts.append(separator)
                parts.append(self._format_selector_msg(explanation, selectors))
        return ' '.join(parts)

    def _format_selector_msg(self, explanation, selectors):
        if False:
            for i in range(10):
                print('nop')
        if len(selectors) == 1 and explanation[-1] == 's':
            explanation = explanation[:-1]
        return f"{explanation} {seq2str(selectors, lastsep=' or ')}"

    def _get_suite_selector_msg(self):
        if False:
            return 10
        if not self.include_suites:
            return ''
        return self._format_selector_msg('in suites', self.include_suites)