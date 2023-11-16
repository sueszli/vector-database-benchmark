"""Report test results in JUnit-XML format, for use with Jenkins and build
integration servers.

Based on initial code from Ross Lawley.

Output conforms to
https://github.com/jenkinsci/xunit-plugin/blob/master/src/main/resources/org/jenkinsci/plugins/xunit/types/model/xsd/junit-10.xsd
"""
import functools
import os
import platform
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import pytest
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
xml_key = StashKey['LogXML']()

def bin_xml_escape(arg: object) -> str:
    if False:
        while True:
            i = 10
    "Visually escape invalid XML characters.\n\n    For example, transforms\n        'hello\\aworld\\b'\n    into\n        'hello#x07world#x08'\n    Note that the #xABs are *not* XML escapes - missing the ampersand &#xAB.\n    The idea is to escape visually for the user rather than for XML itself.\n    "

    def repl(matchobj: Match[str]) -> str:
        if False:
            while True:
                i = 10
        i = ord(matchobj.group())
        if i <= 255:
            return '#x%02X' % i
        else:
            return '#x%04X' % i
    illegal_xml_re = '[^\t\n\r -~\x80-\ud7ff\ue000-�က0-ჿFF]'
    return re.sub(illegal_xml_re, repl, str(arg))

def merge_family(left, right) -> None:
    if False:
        while True:
            i = 10
    result = {}
    for (kl, vl) in left.items():
        for (kr, vr) in right.items():
            if not isinstance(vl, list):
                raise TypeError(type(vl))
            result[kl] = vl + vr
    left.update(result)
families = {}
families['_base'] = {'testcase': ['classname', 'name']}
families['_base_legacy'] = {'testcase': ['file', 'line', 'url']}
families['xunit1'] = families['_base'].copy()
merge_family(families['xunit1'], families['_base_legacy'])
families['xunit2'] = families['_base']

class _NodeReporter:

    def __init__(self, nodeid: Union[str, TestReport], xml: 'LogXML') -> None:
        if False:
            i = 10
            return i + 15
        self.id = nodeid
        self.xml = xml
        self.add_stats = self.xml.add_stats
        self.family = self.xml.family
        self.duration = 0.0
        self.properties: List[Tuple[str, str]] = []
        self.nodes: List[ET.Element] = []
        self.attrs: Dict[str, str] = {}

    def append(self, node: ET.Element) -> None:
        if False:
            return 10
        self.xml.add_stats(node.tag)
        self.nodes.append(node)

    def add_property(self, name: str, value: object) -> None:
        if False:
            return 10
        self.properties.append((str(name), bin_xml_escape(value)))

    def add_attribute(self, name: str, value: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.attrs[str(name)] = bin_xml_escape(value)

    def make_properties_node(self) -> Optional[ET.Element]:
        if False:
            i = 10
            return i + 15
        'Return a Junit node containing custom properties, if any.'
        if self.properties:
            properties = ET.Element('properties')
            for (name, value) in self.properties:
                properties.append(ET.Element('property', name=name, value=value))
            return properties
        return None

    def record_testreport(self, testreport: TestReport) -> None:
        if False:
            print('Hello World!')
        names = mangle_test_address(testreport.nodeid)
        existing_attrs = self.attrs
        classnames = names[:-1]
        if self.xml.prefix:
            classnames.insert(0, self.xml.prefix)
        attrs: Dict[str, str] = {'classname': '.'.join(classnames), 'name': bin_xml_escape(names[-1]), 'file': testreport.location[0]}
        if testreport.location[1] is not None:
            attrs['line'] = str(testreport.location[1])
        if hasattr(testreport, 'url'):
            attrs['url'] = testreport.url
        self.attrs = attrs
        self.attrs.update(existing_attrs)
        if self.family == 'xunit1':
            return
        temp_attrs = {}
        for key in self.attrs.keys():
            if key in families[self.family]['testcase']:
                temp_attrs[key] = self.attrs[key]
        self.attrs = temp_attrs

    def to_xml(self) -> ET.Element:
        if False:
            print('Hello World!')
        testcase = ET.Element('testcase', self.attrs, time='%.3f' % self.duration)
        properties = self.make_properties_node()
        if properties is not None:
            testcase.append(properties)
        testcase.extend(self.nodes)
        return testcase

    def _add_simple(self, tag: str, message: str, data: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        node = ET.Element(tag, message=message)
        node.text = bin_xml_escape(data)
        self.append(node)

    def write_captured_output(self, report: TestReport) -> None:
        if False:
            while True:
                i = 10
        if not self.xml.log_passing_tests and report.passed:
            return
        content_out = report.capstdout
        content_log = report.caplog
        content_err = report.capstderr
        if self.xml.logging == 'no':
            return
        content_all = ''
        if self.xml.logging in ['log', 'all']:
            content_all = self._prepare_content(content_log, ' Captured Log ')
        if self.xml.logging in ['system-out', 'out-err', 'all']:
            content_all += self._prepare_content(content_out, ' Captured Out ')
            self._write_content(report, content_all, 'system-out')
            content_all = ''
        if self.xml.logging in ['system-err', 'out-err', 'all']:
            content_all += self._prepare_content(content_err, ' Captured Err ')
            self._write_content(report, content_all, 'system-err')
            content_all = ''
        if content_all:
            self._write_content(report, content_all, 'system-out')

    def _prepare_content(self, content: str, header: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join([header.center(80, '-'), content, ''])

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        tag = ET.Element(jheader)
        tag.text = bin_xml_escape(content)
        self.append(tag)

    def append_pass(self, report: TestReport) -> None:
        if False:
            i = 10
            return i + 15
        self.add_stats('passed')

    def append_failure(self, report: TestReport) -> None:
        if False:
            for i in range(10):
                print('nop')
        if hasattr(report, 'wasxfail'):
            self._add_simple('skipped', 'xfail-marked test passes unexpectedly')
        else:
            assert report.longrepr is not None
            reprcrash: Optional[ReprFileLocation] = getattr(report.longrepr, 'reprcrash', None)
            if reprcrash is not None:
                message = reprcrash.message
            else:
                message = str(report.longrepr)
            message = bin_xml_escape(message)
            self._add_simple('failure', message, str(report.longrepr))

    def append_collect_error(self, report: TestReport) -> None:
        if False:
            return 10
        assert report.longrepr is not None
        self._add_simple('error', 'collection failure', str(report.longrepr))

    def append_collect_skipped(self, report: TestReport) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._add_simple('skipped', 'collection skipped', str(report.longrepr))

    def append_error(self, report: TestReport) -> None:
        if False:
            print('Hello World!')
        assert report.longrepr is not None
        reprcrash: Optional[ReprFileLocation] = getattr(report.longrepr, 'reprcrash', None)
        if reprcrash is not None:
            reason = reprcrash.message
        else:
            reason = str(report.longrepr)
        if report.when == 'teardown':
            msg = f'failed on teardown with "{reason}"'
        else:
            msg = f'failed on setup with "{reason}"'
        self._add_simple('error', bin_xml_escape(msg), str(report.longrepr))

    def append_skipped(self, report: TestReport) -> None:
        if False:
            return 10
        if hasattr(report, 'wasxfail'):
            xfailreason = report.wasxfail
            if xfailreason.startswith('reason: '):
                xfailreason = xfailreason[8:]
            xfailreason = bin_xml_escape(xfailreason)
            skipped = ET.Element('skipped', type='pytest.xfail', message=xfailreason)
            self.append(skipped)
        else:
            assert isinstance(report.longrepr, tuple)
            (filename, lineno, skipreason) = report.longrepr
            if skipreason.startswith('Skipped: '):
                skipreason = skipreason[9:]
            details = f'{filename}:{lineno}: {skipreason}'
            skipped = ET.Element('skipped', type='pytest.skip', message=skipreason)
            skipped.text = bin_xml_escape(details)
            self.append(skipped)
            self.write_captured_output(report)

    def finalize(self) -> None:
        if False:
            print('Hello World!')
        data = self.to_xml()
        self.__dict__.clear()
        self.to_xml = lambda : data

def _warn_incompatibility_with_xunit2(request: FixtureRequest, fixture_name: str) -> None:
    if False:
        return 10
    'Emit a PytestWarning about the given fixture being incompatible with newer xunit revisions.'
    from _pytest.warning_types import PytestWarning
    xml = request.config.stash.get(xml_key, None)
    if xml is not None and xml.family not in ('xunit1', 'legacy'):
        request.node.warn(PytestWarning("{fixture_name} is incompatible with junit_family '{family}' (use 'legacy' or 'xunit1')".format(fixture_name=fixture_name, family=xml.family)))

@pytest.fixture
def record_property(request: FixtureRequest) -> Callable[[str, object], None]:
    if False:
        return 10
    'Add extra properties to the calling test.\n\n    User properties become part of the test report and are available to the\n    configured reporters, like JUnit XML.\n\n    The fixture is callable with ``name, value``. The value is automatically\n    XML-encoded.\n\n    Example::\n\n        def test_function(record_property):\n            record_property("example_key", 1)\n    '
    _warn_incompatibility_with_xunit2(request, 'record_property')

    def append_property(name: str, value: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        request.node.user_properties.append((name, value))
    return append_property

@pytest.fixture
def record_xml_attribute(request: FixtureRequest) -> Callable[[str, object], None]:
    if False:
        print('Hello World!')
    'Add extra xml attributes to the tag for the calling test.\n\n    The fixture is callable with ``name, value``. The value is\n    automatically XML-encoded.\n    '
    from _pytest.warning_types import PytestExperimentalApiWarning
    request.node.warn(PytestExperimentalApiWarning('record_xml_attribute is an experimental feature'))
    _warn_incompatibility_with_xunit2(request, 'record_xml_attribute')

    def add_attr_noop(name: str, value: object) -> None:
        if False:
            print('Hello World!')
        pass
    attr_func = add_attr_noop
    xml = request.config.stash.get(xml_key, None)
    if xml is not None:
        node_reporter = xml.node_reporter(request.node.nodeid)
        attr_func = node_reporter.add_attribute
    return attr_func

def _check_record_param_type(param: str, v: str) -> None:
    if False:
        return 10
    'Used by record_testsuite_property to check that the given parameter name is of the proper\n    type.'
    __tracebackhide__ = True
    if not isinstance(v, str):
        msg = '{param} parameter needs to be a string, but {g} given'
        raise TypeError(msg.format(param=param, g=type(v).__name__))

@pytest.fixture(scope='session')
def record_testsuite_property(request: FixtureRequest) -> Callable[[str, object], None]:
    if False:
        for i in range(10):
            print('nop')
    'Record a new ``<property>`` tag as child of the root ``<testsuite>``.\n\n    This is suitable to writing global information regarding the entire test\n    suite, and is compatible with ``xunit2`` JUnit family.\n\n    This is a ``session``-scoped fixture which is called with ``(name, value)``. Example:\n\n    .. code-block:: python\n\n        def test_foo(record_testsuite_property):\n            record_testsuite_property("ARCH", "PPC")\n            record_testsuite_property("STORAGE_TYPE", "CEPH")\n\n    :param name:\n        The property name.\n    :param value:\n        The property value. Will be converted to a string.\n\n    .. warning::\n\n        Currently this fixture **does not work** with the\n        `pytest-xdist <https://github.com/pytest-dev/pytest-xdist>`__ plugin. See\n        :issue:`7767` for details.\n    '
    __tracebackhide__ = True

    def record_func(name: str, value: object) -> None:
        if False:
            while True:
                i = 10
        'No-op function in case --junit-xml was not passed in the command-line.'
        __tracebackhide__ = True
        _check_record_param_type('name', name)
    xml = request.config.stash.get(xml_key, None)
    if xml is not None:
        record_func = xml.add_global_property
    return record_func

def pytest_addoption(parser: Parser) -> None:
    if False:
        print('Hello World!')
    group = parser.getgroup('terminal reporting')
    group.addoption('--junitxml', '--junit-xml', action='store', dest='xmlpath', metavar='path', type=functools.partial(filename_arg, optname='--junitxml'), default=None, help='Create junit-xml style report file at given path')
    group.addoption('--junitprefix', '--junit-prefix', action='store', metavar='str', default=None, help='Prepend prefix to classnames in junit-xml output')
    parser.addini('junit_suite_name', 'Test suite name for JUnit report', default='pytest')
    parser.addini('junit_logging', 'Write captured log messages to JUnit report: one of no|log|system-out|system-err|out-err|all', default='no')
    parser.addini('junit_log_passing_tests', 'Capture log information for passing tests to JUnit report: ', type='bool', default=True)
    parser.addini('junit_duration_report', 'Duration time to report: one of total|call', default='total')
    parser.addini('junit_family', 'Emit XML for schema: one of legacy|xunit1|xunit2', default='xunit2')

def pytest_configure(config: Config) -> None:
    if False:
        while True:
            i = 10
    xmlpath = config.option.xmlpath
    if xmlpath and (not hasattr(config, 'workerinput')):
        junit_family = config.getini('junit_family')
        config.stash[xml_key] = LogXML(xmlpath, config.option.junitprefix, config.getini('junit_suite_name'), config.getini('junit_logging'), config.getini('junit_duration_report'), junit_family, config.getini('junit_log_passing_tests'))
        config.pluginmanager.register(config.stash[xml_key])

def pytest_unconfigure(config: Config) -> None:
    if False:
        for i in range(10):
            print('nop')
    xml = config.stash.get(xml_key, None)
    if xml:
        del config.stash[xml_key]
        config.pluginmanager.unregister(xml)

def mangle_test_address(address: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    (path, possible_open_bracket, params) = address.partition('[')
    names = path.split('::')
    names[0] = names[0].replace(nodes.SEP, '.')
    names[0] = re.sub('\\.py$', '', names[0])
    names[-1] += possible_open_bracket + params
    return names

class LogXML:

    def __init__(self, logfile, prefix: Optional[str], suite_name: str='pytest', logging: str='no', report_duration: str='total', family='xunit1', log_passing_tests: bool=True) -> None:
        if False:
            print('Hello World!')
        logfile = os.path.expanduser(os.path.expandvars(logfile))
        self.logfile = os.path.normpath(os.path.abspath(logfile))
        self.prefix = prefix
        self.suite_name = suite_name
        self.logging = logging
        self.log_passing_tests = log_passing_tests
        self.report_duration = report_duration
        self.family = family
        self.stats: Dict[str, int] = dict.fromkeys(['error', 'passed', 'failure', 'skipped'], 0)
        self.node_reporters: Dict[Tuple[Union[str, TestReport], object], _NodeReporter] = {}
        self.node_reporters_ordered: List[_NodeReporter] = []
        self.global_properties: List[Tuple[str, str]] = []
        self.open_reports: List[TestReport] = []
        self.cnt_double_fail_tests = 0
        if self.family == 'legacy':
            self.family = 'xunit1'

    def finalize(self, report: TestReport) -> None:
        if False:
            return 10
        nodeid = getattr(report, 'nodeid', report)
        workernode = getattr(report, 'node', None)
        reporter = self.node_reporters.pop((nodeid, workernode))
        for (propname, propvalue) in report.user_properties:
            reporter.add_property(propname, str(propvalue))
        if reporter is not None:
            reporter.finalize()

    def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporter:
        if False:
            print('Hello World!')
        nodeid: Union[str, TestReport] = getattr(report, 'nodeid', report)
        workernode = getattr(report, 'node', None)
        key = (nodeid, workernode)
        if key in self.node_reporters:
            return self.node_reporters[key]
        reporter = _NodeReporter(nodeid, self)
        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)
        return reporter

    def add_stats(self, key: str) -> None:
        if False:
            print('Hello World!')
        if key in self.stats:
            self.stats[key] += 1

    def _opentestcase(self, report: TestReport) -> _NodeReporter:
        if False:
            print('Hello World!')
        reporter = self.node_reporter(report)
        reporter.record_testreport(report)
        return reporter

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if False:
            while True:
                i = 10
        'Handle a setup/call/teardown report, generating the appropriate\n        XML tags as necessary.\n\n        Note: due to plugins like xdist, this hook may be called in interlaced\n        order with reports from other nodes. For example:\n\n        Usual call order:\n            -> setup node1\n            -> call node1\n            -> teardown node1\n            -> setup node2\n            -> call node2\n            -> teardown node2\n\n        Possible call order in xdist:\n            -> setup node1\n            -> call node1\n            -> setup node2\n            -> call node2\n            -> teardown node2\n            -> teardown node1\n        '
        close_report = None
        if report.passed:
            if report.when == 'call':
                reporter = self._opentestcase(report)
                reporter.append_pass(report)
        elif report.failed:
            if report.when == 'teardown':
                report_wid = getattr(report, 'worker_id', None)
                report_ii = getattr(report, 'item_index', None)
                close_report = next((rep for rep in self.open_reports if rep.nodeid == report.nodeid and getattr(rep, 'item_index', None) == report_ii and (getattr(rep, 'worker_id', None) == report_wid)), None)
                if close_report:
                    self.finalize(close_report)
                    self.cnt_double_fail_tests += 1
            reporter = self._opentestcase(report)
            if report.when == 'call':
                reporter.append_failure(report)
                self.open_reports.append(report)
                if not self.log_passing_tests:
                    reporter.write_captured_output(report)
            else:
                reporter.append_error(report)
        elif report.skipped:
            reporter = self._opentestcase(report)
            reporter.append_skipped(report)
        self.update_testcase_duration(report)
        if report.when == 'teardown':
            reporter = self._opentestcase(report)
            reporter.write_captured_output(report)
            self.finalize(report)
            report_wid = getattr(report, 'worker_id', None)
            report_ii = getattr(report, 'item_index', None)
            close_report = next((rep for rep in self.open_reports if rep.nodeid == report.nodeid and getattr(rep, 'item_index', None) == report_ii and (getattr(rep, 'worker_id', None) == report_wid)), None)
            if close_report:
                self.open_reports.remove(close_report)

    def update_testcase_duration(self, report: TestReport) -> None:
        if False:
            i = 10
            return i + 15
        'Accumulate total duration for nodeid from given report and update\n        the Junit.testcase with the new total if already created.'
        if self.report_duration == 'total' or report.when == self.report_duration:
            reporter = self.node_reporter(report)
            reporter.duration += getattr(report, 'duration', 0.0)

    def pytest_collectreport(self, report: TestReport) -> None:
        if False:
            print('Hello World!')
        if not report.passed:
            reporter = self._opentestcase(report)
            if report.failed:
                reporter.append_collect_error(report)
            else:
                reporter.append_collect_skipped(report)

    def pytest_internalerror(self, excrepr: ExceptionRepr) -> None:
        if False:
            i = 10
            return i + 15
        reporter = self.node_reporter('internal')
        reporter.attrs.update(classname='pytest', name='internal')
        reporter._add_simple('error', 'internal error', str(excrepr))

    def pytest_sessionstart(self) -> None:
        if False:
            return 10
        self.suite_start_time = timing.time()

    def pytest_sessionfinish(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        dirname = os.path.dirname(os.path.abspath(self.logfile))
        os.makedirs(dirname, exist_ok=True)
        with open(self.logfile, 'w', encoding='utf-8') as logfile:
            suite_stop_time = timing.time()
            suite_time_delta = suite_stop_time - self.suite_start_time
            numtests = self.stats['passed'] + self.stats['failure'] + self.stats['skipped'] + self.stats['error'] - self.cnt_double_fail_tests
            logfile.write('<?xml version="1.0" encoding="utf-8"?>')
            suite_node = ET.Element('testsuite', name=self.suite_name, errors=str(self.stats['error']), failures=str(self.stats['failure']), skipped=str(self.stats['skipped']), tests=str(numtests), time='%.3f' % suite_time_delta, timestamp=datetime.fromtimestamp(self.suite_start_time).isoformat(), hostname=platform.node())
            global_properties = self._get_global_properties_node()
            if global_properties is not None:
                suite_node.append(global_properties)
            for node_reporter in self.node_reporters_ordered:
                suite_node.append(node_reporter.to_xml())
            testsuites = ET.Element('testsuites')
            testsuites.append(suite_node)
            logfile.write(ET.tostring(testsuites, encoding='unicode'))

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        if False:
            for i in range(10):
                print('nop')
        terminalreporter.write_sep('-', f'generated xml file: {self.logfile}')

    def add_global_property(self, name: str, value: object) -> None:
        if False:
            print('Hello World!')
        __tracebackhide__ = True
        _check_record_param_type('name', name)
        self.global_properties.append((name, bin_xml_escape(value)))

    def _get_global_properties_node(self) -> Optional[ET.Element]:
        if False:
            print('Hello World!')
        'Return a Junit node containing custom properties, if any.'
        if self.global_properties:
            properties = ET.Element('properties')
            for (name, value) in self.global_properties:
                properties.append(ET.Element('property', name=name, value=value))
            return properties
        return None