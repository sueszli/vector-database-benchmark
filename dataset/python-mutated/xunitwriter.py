from robot.result import ResultVisitor, TestCase, TestSuite
from robot.utils import XmlWriter

class XUnitWriter:

    def __init__(self, execution_result):
        if False:
            i = 10
            return i + 15
        self._execution_result = execution_result

    def write(self, output):
        if False:
            i = 10
            return i + 15
        xml_writer = XmlWriter(output, usage='xunit')
        writer = XUnitFileWriter(xml_writer)
        self._execution_result.visit(writer)

class XUnitFileWriter(ResultVisitor):
    """Provides an xUnit-compatible result file.

    Attempts to adhere to the de facto schema guessed by Peter Reilly, see:
    http://marc.info/?l=ant-dev&m=123551933508682
    """

    def __init__(self, xml_writer):
        if False:
            return 10
        self._writer = xml_writer

    def start_suite(self, suite: TestSuite):
        if False:
            while True:
                i = 10
        stats = suite.statistics
        attrs = {'name': suite.name, 'tests': str(stats.total), 'errors': '0', 'failures': str(stats.failed), 'skipped': str(stats.skipped), 'time': format(suite.elapsed_time.total_seconds(), '.3f'), 'timestamp': suite.start_time.isoformat() if suite.start_time else None}
        self._writer.start('testsuite', attrs)

    def end_suite(self, suite: TestSuite):
        if False:
            i = 10
            return i + 15
        if suite.metadata or suite.doc:
            self._writer.start('properties')
            if suite.doc:
                self._writer.element('property', attrs={'name': 'Documentation', 'value': suite.doc})
            for (meta_name, meta_value) in suite.metadata.items():
                self._writer.element('property', attrs={'name': meta_name, 'value': meta_value})
            self._writer.end('properties')
        self._writer.end('testsuite')

    def visit_test(self, test: TestCase):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('testcase', {'classname': test.parent.full_name, 'name': test.name, 'time': format(test.elapsed_time.total_seconds(), '.3f')})
        if test.failed:
            self._writer.element('failure', attrs={'message': test.message, 'type': 'AssertionError'})
        if test.skipped:
            self._writer.element('skipped', attrs={'message': test.message, 'type': 'SkipExecution'})
        self._writer.end('testcase')

    def visit_keyword(self, kw):
        if False:
            print('Hello World!')
        pass

    def visit_statistics(self, stats):
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_errors(self, errors):
        if False:
            i = 10
            return i + 15
        pass

    def end_result(self, result):
        if False:
            print('Hello World!')
        self._writer.close()