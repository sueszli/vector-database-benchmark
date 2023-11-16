"""Module implementing the command line entry point for the `Testdoc` tool.

This module can be executed from the command line using the following
approaches::

    python -m robot.testdoc
    python path/to/robot/testdoc.py

Instead of ``python`` it is possible to use also other Python interpreters.

This module also provides :func:`testdoc` and :func:`testdoc_cli` functions
that can be used programmatically. Other code is for internal usage.
"""
import sys
import time
from pathlib import Path
if __name__ == '__main__' and 'robot' not in sys.modules:
    import pythonpathsetter
from robot.conf import RobotSettings
from robot.htmldata import HtmlFileWriter, ModelWriter, JsonWriter, TESTDOC
from robot.running import TestSuiteBuilder
from robot.utils import abspath, Application, file_writer, get_link_path, html_escape, html_format, is_list_like, secs_to_timestr, seq2str2, timestr_to_secs, unescape
USAGE = 'robot.testdoc -- Robot Framework test data documentation tool\n\nVersion:  <VERSION>\n\nUsage:  python -m robot.testdoc [options] data_sources output_file\n\nTestdoc generates a high level test documentation based on Robot Framework\ntest data. Generated documentation includes name, documentation and other\nmetadata of each test suite and test case, as well as the top-level keywords\nand their arguments.\n\nOptions\n=======\n\n  -T --title title       Set the title of the generated documentation.\n                         Underscores in the title are converted to spaces.\n                         The default title is the name of the top level suite.\n  -N --name name         Override the name of the top level suite.\n  -D --doc document      Override the documentation of the top level suite.\n  -M --metadata name:value *  Set/override metadata of the top level suite.\n  -G --settag tag *      Set given tag(s) to all test cases.\n  -t --test name *       Include tests by name.\n  -s --suite name *      Include suites by name.\n  -i --include tag *     Include tests by tags.\n  -e --exclude tag *     Exclude tests by tags.\n  -A --argumentfile path *  Text file to read more arguments from. Use special\n                          path `STDIN` to read contents from the standard input\n                          stream. File can have both options and data sources\n                          one per line. Contents do not need to be escaped but\n                          spaces in the beginning and end of lines are removed.\n                          Empty lines and lines starting with a hash character\n                          (#) are ignored.\n                          Example file:\n                          |  --name Example\n                          |  # This is a comment line\n                          |  my_tests.robot\n                          |  output.html\n                          Examples:\n                          --argumentfile argfile.txt --argumentfile STDIN\n  -h -? --help           Print this help.\n\nAll options except --title have exactly same semantics as same options have\nwhen executing test cases.\n\nExecution\n=========\n\nData can be given as a single file, directory, or as multiple files and\ndirectories. In all these cases, the last argument must be the file where\nto write the output. The output is always created in HTML format.\n\nTestdoc works with all interpreters supported by Robot Framework.\nIt can be executed as an installed module like\n`python -m robot.testdoc` or as a script like `python path/robot/testdoc.py`.\n\nExamples:\n\n  python -m robot.testdoc my_test.robot testdoc.html\n  python path/to/robot/testdoc.py first_suite.txt second_suite.txt output.html\n\nFor more information about Testdoc and other built-in tools, see\nhttp://robotframework.org/robotframework/#built-in-tools.\n'

class TestDoc(Application):

    def __init__(self):
        if False:
            while True:
                i = 10
        Application.__init__(self, USAGE, arg_limits=(2,))

    def main(self, datasources, title=None, **options):
        if False:
            print('Hello World!')
        outfile = abspath(datasources.pop())
        suite = TestSuiteFactory(datasources, **options)
        self._write_test_doc(suite, outfile, title)
        self.console(outfile)

    def _write_test_doc(self, suite, outfile, title):
        if False:
            for i in range(10):
                print('nop')
        with file_writer(outfile, usage='Testdoc output') as output:
            model_writer = TestdocModelWriter(output, suite, title)
            HtmlFileWriter(output, model_writer).write(TESTDOC)

def TestSuiteFactory(datasources, **options):
    if False:
        return 10
    settings = RobotSettings(options)
    if not is_list_like(datasources):
        datasources = [datasources]
    suite = TestSuiteBuilder(process_curdir=False).build(*datasources)
    suite.configure(**settings.suite_config)
    return suite

class TestdocModelWriter(ModelWriter):

    def __init__(self, output, suite, title=None):
        if False:
            print('Hello World!')
        self._output = output
        self._output_path = getattr(output, 'name', None)
        self._suite = suite
        self._title = title.replace('_', ' ') if title else suite.name

    def write(self, line):
        if False:
            while True:
                i = 10
        self._output.write('<script type="text/javascript">\n')
        self.write_data()
        self._output.write('</script>\n')

    def write_data(self):
        if False:
            i = 10
            return i + 15
        model = {'suite': JsonConverter(self._output_path).convert(self._suite), 'title': self._title, 'generated': int(time.time() * 1000)}
        JsonWriter(self._output).write_json('testdoc = ', model)

class JsonConverter:

    def __init__(self, output_path=None):
        if False:
            while True:
                i = 10
        self._output_path = output_path

    def convert(self, suite):
        if False:
            i = 10
            return i + 15
        return self._convert_suite(suite)

    def _convert_suite(self, suite):
        if False:
            i = 10
            return i + 15
        return {'source': str(suite.source or ''), 'relativeSource': self._get_relative_source(suite.source), 'id': suite.id, 'name': self._escape(suite.name), 'fullName': self._escape(suite.full_name), 'doc': self._html(suite.doc), 'metadata': [(self._escape(name), self._html(value)) for (name, value) in suite.metadata.items()], 'numberOfTests': suite.test_count, 'suites': self._convert_suites(suite), 'tests': self._convert_tests(suite), 'keywords': list(self._convert_keywords((suite.setup, suite.teardown)))}

    def _get_relative_source(self, source):
        if False:
            print('Hello World!')
        if not source or not self._output_path:
            return ''
        return get_link_path(source, Path(self._output_path).parent)

    def _escape(self, item):
        if False:
            for i in range(10):
                print('nop')
        return html_escape(item)

    def _html(self, item):
        if False:
            i = 10
            return i + 15
        return html_format(unescape(item))

    def _convert_suites(self, suite):
        if False:
            i = 10
            return i + 15
        return [self._convert_suite(s) for s in suite.suites]

    def _convert_tests(self, suite):
        if False:
            return 10
        return [self._convert_test(t) for t in suite.tests]

    def _convert_test(self, test):
        if False:
            for i in range(10):
                print('nop')
        if test.setup:
            test.body.insert(0, test.setup)
        if test.teardown:
            test.body.append(test.teardown)
        return {'name': self._escape(test.name), 'fullName': self._escape(test.full_name), 'id': test.id, 'doc': self._html(test.doc), 'tags': [self._escape(t) for t in test.tags], 'timeout': self._get_timeout(test.timeout), 'keywords': list(self._convert_keywords(test.body))}

    def _convert_keywords(self, keywords):
        if False:
            print('Hello World!')
        for kw in keywords:
            if not kw:
                continue
            if kw.type in kw.KEYWORD_TYPES:
                yield self._convert_keyword(kw)
            elif kw.type == kw.FOR:
                yield self._convert_for(kw)
            elif kw.type == kw.WHILE:
                yield self._convert_while(kw)
            elif kw.type == kw.IF_ELSE_ROOT:
                yield from self._convert_if(kw)
            elif kw.type == kw.TRY_EXCEPT_ROOT:
                yield from self._convert_try(kw)
            elif kw.type == kw.VAR:
                yield self._convert_var(kw)

    def _convert_for(self, data):
        if False:
            for i in range(10):
                print('nop')
        name = '%s %s %s' % (', '.join(data.assign), data.flavor, seq2str2(data.values))
        return {'type': 'FOR', 'name': self._escape(name), 'arguments': ''}

    def _convert_while(self, data):
        if False:
            print('Hello World!')
        return {'type': 'WHILE', 'name': self._escape(data.condition), 'arguments': ''}

    def _convert_if(self, data):
        if False:
            return 10
        for branch in data.body:
            yield {'type': branch.type, 'name': self._escape(branch.condition or ''), 'arguments': ''}

    def _convert_try(self, data):
        if False:
            i = 10
            return i + 15
        for branch in data.body:
            if branch.type == branch.EXCEPT:
                patterns = ', '.join(branch.patterns)
                as_var = f'AS {branch.assign}' if branch.assign else ''
                name = f'{patterns} {as_var}'.strip()
            else:
                name = ''
            yield {'type': branch.type, 'name': name, 'arguments': ''}

    def _convert_var(self, data):
        if False:
            return 10
        if data.name[0] == '$' and len(data.value) == 1:
            value = data.value[0]
        else:
            value = '[' + ', '.join(data.value) + ']'
        return {'type': 'VAR', 'name': f'{data.name} = {value}'}

    def _convert_keyword(self, kw):
        if False:
            for i in range(10):
                print('nop')
        return {'type': kw.type, 'name': self._escape(self._get_kw_name(kw)), 'arguments': self._escape(', '.join(kw.args))}

    def _get_kw_name(self, kw):
        if False:
            i = 10
            return i + 15
        if kw.assign:
            return '%s = %s' % (', '.join((a.rstrip('= ') for a in kw.assign)), kw.name)
        return kw.name

    def _get_timeout(self, timeout):
        if False:
            while True:
                i = 10
        if timeout is None:
            return ''
        try:
            tout = secs_to_timestr(timestr_to_secs(timeout))
        except ValueError:
            tout = timeout
        return tout

def testdoc_cli(arguments):
    if False:
        print('Hello World!')
    "Executes `Testdoc` similarly as from the command line.\n\n    :param arguments: command line arguments as a list of strings.\n\n    For programmatic usage the :func:`testdoc` function is typically better. It\n    has a better API for that and does not call :func:`sys.exit` like\n    this function.\n\n    Example::\n\n        from robot.testdoc import testdoc_cli\n\n        testdoc_cli(['--title', 'Test Plan', 'mytests', 'plan.html'])\n    "
    TestDoc().execute_cli(arguments)

def testdoc(*arguments, **options):
    if False:
        i = 10
        return i + 15
    "Executes `Testdoc` programmatically.\n\n    Arguments and options have same semantics, and options have same names,\n    as arguments and options to Testdoc.\n\n    Example::\n\n        from robot.testdoc import testdoc\n\n        testdoc('mytests', 'plan.html', title='Test Plan')\n    "
    TestDoc().execute(*arguments, **options)
if __name__ == '__main__':
    testdoc_cli(sys.argv[1:])