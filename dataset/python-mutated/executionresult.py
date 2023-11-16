from robot.errors import DataError
from robot.model import Statistics
from .executionerrors import ExecutionErrors
from .model import TestSuite

class Result:
    """Test execution results.

    Can be created based on XML output files using the
    :func:`~.resultbuilder.ExecutionResult`
    factory method. Also returned by the
    :meth:`robot.running.TestSuite.run <robot.running.model.TestSuite.run>`
    method.
    """

    def __init__(self, source=None, root_suite=None, errors=None, rpa=None):
        if False:
            while True:
                i = 10
        self.source = source
        self.suite = root_suite or TestSuite()
        self.errors = errors or ExecutionErrors()
        self.generated_by_robot = True
        self._status_rc = True
        self._stat_config = {}
        self.rpa = rpa

    @property
    def statistics(self):
        if False:
            return 10
        "Test execution statistics.\n\n        Statistics are an instance of\n        :class:`~robot.model.statistics.Statistics` that is created based\n        on the contained ``suite`` and possible\n        :func:`configuration <configure>`.\n\n        Statistics are created every time this property is accessed. Saving\n        them to a variable is thus often a good idea to avoid re-creating\n        them unnecessarily::\n\n            from robot.api import ExecutionResult\n\n            result = ExecutionResult('output.xml')\n            result.configure(stat_config={'suite_stat_level': 2,\n                                          'tag_stat_combine': 'tagANDanother'})\n            stats = result.statistics\n            print(stats.total.failed)\n            print(stats.total.passed)\n            print(stats.tags.combined[0].total)\n        "
        return Statistics(self.suite, rpa=self.rpa, **self._stat_config)

    @property
    def return_code(self):
        if False:
            for i in range(10):
                print('nop')
        'Return code (integer) of test execution.\n\n        By default returns the number of failed tests (max 250),\n        but can be :func:`configured <configure>` to always return 0.\n        '
        if self._status_rc:
            return min(self.suite.statistics.failed, 250)
        return 0

    def configure(self, status_rc=True, suite_config=None, stat_config=None):
        if False:
            print('Hello World!')
        'Configures the result object and objects it contains.\n\n        :param status_rc: If set to ``False``, :attr:`return_code` always\n            returns 0.\n        :param suite_config: A dictionary of configuration options passed\n            to :meth:`~.result.testsuite.TestSuite.configure` method of\n            the contained ``suite``.\n        :param stat_config: A dictionary of configuration options used when\n            creating :attr:`statistics`.\n        '
        if suite_config:
            self.suite.configure(**suite_config)
        self._status_rc = status_rc
        self._stat_config = stat_config or {}

    def save(self, path=None):
        if False:
            for i in range(10):
                print('nop')
        'Save results as a new output XML file.\n\n        :param path: Path to save results to. If omitted, overwrites the\n            original file.\n        '
        from robot.reporting.outputwriter import OutputWriter
        self.visit(OutputWriter(path or self.source, rpa=self.rpa))

    def visit(self, visitor):
        if False:
            print('Hello World!')
        'An entry point to visit the whole result object.\n\n        :param visitor: An instance of :class:`~.visitor.ResultVisitor`.\n\n        Visitors can gather information, modify results, etc. See\n        :mod:`~robot.result` package for a simple usage example.\n\n        Notice that it is also possible to call :meth:`result.suite.visit\n        <robot.result.testsuite.TestSuite.visit>` if there is no need to\n        visit the contained ``statistics`` or ``errors``.\n        '
        visitor.visit_result(self)

    def handle_suite_teardown_failures(self):
        if False:
            i = 10
            return i + 15
        'Internal usage only.'
        if self.generated_by_robot:
            self.suite.handle_suite_teardown_failures()

    def set_execution_mode(self, other):
        if False:
            print('Hello World!')
        'Set execution mode based on other result. Internal usage only.'
        if other.rpa is None:
            pass
        elif self.rpa is None:
            self.rpa = other.rpa
        elif self.rpa is not other.rpa:
            (this, that) = ('task', 'test') if other.rpa else ('test', 'task')
            raise DataError("Conflicting execution modes. File '%s' has %ss but files parsed earlier have %ss. Use '--rpa' or '--norpa' options to set the execution mode explicitly." % (other.source, this, that))

class CombinedResult(Result):
    """Combined results of multiple test executions."""

    def __init__(self, results=None):
        if False:
            for i in range(10):
                print('nop')
        Result.__init__(self)
        for result in results or ():
            self.add_result(result)

    def add_result(self, other):
        if False:
            return 10
        self.set_execution_mode(other)
        self.suite.suites.append(other.suite)
        self.errors.add(other.errors)