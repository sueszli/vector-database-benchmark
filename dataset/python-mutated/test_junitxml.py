import os
import platform
from datetime import datetime
from pathlib import Path
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from xml.dom import minidom
import xmlschema
import pytest
from _pytest.config import Config
from _pytest.junitxml import bin_xml_escape
from _pytest.junitxml import LogXML
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.pytester import RunResult
from _pytest.reports import BaseReport
from _pytest.reports import TestReport
from _pytest.stash import Stash

@pytest.fixture(scope='session')
def schema() -> xmlschema.XMLSchema:
    if False:
        i = 10
        return i + 15
    'Return an xmlschema.XMLSchema object for the junit-10.xsd file.'
    fn = Path(__file__).parent / 'example_scripts/junit-10.xsd'
    with fn.open(encoding='utf-8') as f:
        return xmlschema.XMLSchema(f)

class RunAndParse:

    def __init__(self, pytester: Pytester, schema: xmlschema.XMLSchema) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pytester = pytester
        self.schema = schema

    def __call__(self, *args: Union[str, 'os.PathLike[str]'], family: Optional[str]='xunit1') -> Tuple[RunResult, 'DomNode']:
        if False:
            print('Hello World!')
        if family:
            args = ('-o', 'junit_family=' + family) + args
        xml_path = self.pytester.path.joinpath('junit.xml')
        result = self.pytester.runpytest('--junitxml=%s' % xml_path, *args)
        if family == 'xunit2':
            with xml_path.open(encoding='utf-8') as f:
                self.schema.validate(f)
        xmldoc = minidom.parse(str(xml_path))
        return (result, DomNode(xmldoc))

@pytest.fixture
def run_and_parse(pytester: Pytester, schema: xmlschema.XMLSchema) -> RunAndParse:
    if False:
        for i in range(10):
            print('nop')
    'Fixture that returns a function that can be used to execute pytest and\n    return the parsed ``DomNode`` of the root xml node.\n\n    The ``family`` parameter is used to configure the ``junit_family`` of the written report.\n    "xunit2" is also automatically validated against the schema.\n    '
    return RunAndParse(pytester, schema)

def assert_attr(node, **kwargs):
    if False:
        while True:
            i = 10
    __tracebackhide__ = True

    def nodeval(node, name):
        if False:
            print('Hello World!')
        anode = node.getAttributeNode(name)
        if anode is not None:
            return anode.value
    expected = {name: str(value) for (name, value) in kwargs.items()}
    on_node = {name: nodeval(node, name) for name in expected}
    assert on_node == expected

class DomNode:

    def __init__(self, dom):
        if False:
            for i in range(10):
                print('nop')
        self.__node = dom

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__node.toxml()

    def find_first_by_tag(self, tag):
        if False:
            i = 10
            return i + 15
        return self.find_nth_by_tag(tag, 0)

    def _by_tag(self, tag):
        if False:
            return 10
        return self.__node.getElementsByTagName(tag)

    @property
    def children(self):
        if False:
            while True:
                i = 10
        return [type(self)(x) for x in self.__node.childNodes]

    @property
    def get_unique_child(self):
        if False:
            return 10
        children = self.children
        assert len(children) == 1
        return children[0]

    def find_nth_by_tag(self, tag, n):
        if False:
            while True:
                i = 10
        items = self._by_tag(tag)
        try:
            nth = items[n]
        except IndexError:
            pass
        else:
            return type(self)(nth)

    def find_by_tag(self, tag):
        if False:
            print('Hello World!')
        t = type(self)
        return [t(x) for x in self.__node.getElementsByTagName(tag)]

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        node = self.__node.getAttributeNode(key)
        if node is not None:
            return node.value

    def assert_attr(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        __tracebackhide__ = True
        return assert_attr(self.__node, **kwargs)

    def toxml(self):
        if False:
            while True:
                i = 10
        return self.__node.toxml()

    @property
    def text(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__node.childNodes[0].wholeText

    @property
    def tag(self):
        if False:
            return 10
        return self.__node.tagName

    @property
    def next_sibling(self):
        if False:
            i = 10
            return i + 15
        return type(self)(self.__node.nextSibling)
parametrize_families = pytest.mark.parametrize('xunit_family', ['xunit1', 'xunit2'])

class TestPython:

    @parametrize_families
    def test_summing_simple(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            def test_pass():\n                pass\n            def test_fail():\n                assert 0\n            def test_skip():\n                pytest.skip("")\n            @pytest.mark.xfail\n            def test_xfail():\n                assert 0\n            @pytest.mark.xfail\n            def test_xpass():\n                assert 1\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(name='pytest', errors=0, failures=1, skipped=2, tests=5)

    @parametrize_families
    def test_summing_simple_with_errors(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            @pytest.fixture\n            def fixture():\n                raise Exception()\n            def test_pass():\n                pass\n            def test_fail():\n                assert 0\n            def test_error(fixture):\n                pass\n            @pytest.mark.xfail\n            def test_xfail():\n                assert False\n            @pytest.mark.xfail(strict=True)\n            def test_xpass():\n                assert True\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(name='pytest', errors=1, failures=2, skipped=1, tests=5)

    @parametrize_families
    def test_hostname_in_xml(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_pass():\n                pass\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(hostname=platform.node())

    @parametrize_families
    def test_timestamp_in_xml(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_pass():\n                pass\n        ')
        start_time = datetime.now()
        (result, dom) = run_and_parse(family=xunit_family)
        node = dom.find_first_by_tag('testsuite')
        timestamp = datetime.strptime(node['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
        assert start_time <= timestamp < datetime.now()

    def test_timing_function(self, pytester: Pytester, run_and_parse: RunAndParse, mock_timing) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            from _pytest import timing\n            def setup_module():\n                timing.sleep(1)\n            def teardown_module():\n                timing.sleep(2)\n            def test_sleep():\n                timing.sleep(4)\n        ')
        (result, dom) = run_and_parse()
        node = dom.find_first_by_tag('testsuite')
        tnode = node.find_first_by_tag('testcase')
        val = tnode['time']
        assert float(val) == 7.0

    @pytest.mark.parametrize('duration_report', ['call', 'total'])
    def test_junit_duration_report(self, pytester: Pytester, monkeypatch: MonkeyPatch, duration_report: str, run_and_parse: RunAndParse) -> None:
        if False:
            return 10
        original_node_reporter = LogXML.node_reporter

        def node_reporter_wrapper(s, report):
            if False:
                i = 10
                return i + 15
            report.duration = 1.0
            reporter = original_node_reporter(s, report)
            return reporter
        monkeypatch.setattr(LogXML, 'node_reporter', node_reporter_wrapper)
        pytester.makepyfile('\n            def test_foo():\n                pass\n        ')
        (result, dom) = run_and_parse('-o', f'junit_duration_report={duration_report}')
        node = dom.find_first_by_tag('testsuite')
        tnode = node.find_first_by_tag('testcase')
        val = float(tnode['time'])
        if duration_report == 'total':
            assert val == 3.0
        else:
            assert duration_report == 'call'
            assert val == 1.0

    @parametrize_families
    def test_setup_error(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture\n            def arg(request):\n                raise ValueError("Error reason")\n            def test_function(arg):\n                pass\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(errors=1, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_setup_error', name='test_function')
        fnode = tnode.find_first_by_tag('error')
        fnode.assert_attr(message='failed on setup with "ValueError: Error reason"')
        assert 'ValueError' in fnode.toxml()

    @parametrize_families
    def test_teardown_error(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture\n            def arg():\n                yield\n                raise ValueError('Error reason')\n            def test_function(arg):\n                pass\n        ")
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_teardown_error', name='test_function')
        fnode = tnode.find_first_by_tag('error')
        fnode.assert_attr(message='failed on teardown with "ValueError: Error reason"')
        assert 'ValueError' in fnode.toxml()

    @parametrize_families
    def test_call_failure_teardown_error(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture\n            def arg():\n                yield\n                raise Exception("Teardown Exception")\n            def test_function(arg):\n                raise Exception("Call Exception")\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(errors=1, failures=1, tests=1)
        (first, second) = dom.find_by_tag('testcase')
        assert first
        assert second
        assert first != second
        fnode = first.find_first_by_tag('failure')
        fnode.assert_attr(message='Exception: Call Exception')
        snode = second.find_first_by_tag('error')
        snode.assert_attr(message='failed on teardown with "Exception: Teardown Exception"')

    @parametrize_families
    def test_skip_contains_name_reason(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            def test_skip():\n                pytest.skip("hello23")\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret == 0
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_skip_contains_name_reason', name='test_skip')
        snode = tnode.find_first_by_tag('skipped')
        snode.assert_attr(type='pytest.skip', message='hello23')

    @parametrize_families
    def test_mark_skip_contains_name_reason(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.skip(reason="hello24")\n            def test_skip():\n                assert True\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret == 0
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_mark_skip_contains_name_reason', name='test_skip')
        snode = tnode.find_first_by_tag('skipped')
        snode.assert_attr(type='pytest.skip', message='hello24')

    @parametrize_families
    def test_mark_skipif_contains_name_reason(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            import pytest\n            GLOBAL_CONDITION = True\n            @pytest.mark.skipif(GLOBAL_CONDITION, reason="hello25")\n            def test_skip():\n                assert True\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret == 0
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_mark_skipif_contains_name_reason', name='test_skip')
        snode = tnode.find_first_by_tag('skipped')
        snode.assert_attr(type='pytest.skip', message='hello25')

    @parametrize_families
    def test_mark_skip_doesnt_capture_output(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.skip(reason="foo")\n            def test_skip():\n                print("bar!")\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret == 0
        node_xml = dom.find_first_by_tag('testsuite').toxml()
        assert 'bar!' not in node_xml

    @parametrize_families
    def test_classname_instance(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            class TestClass(object):\n                def test_method(self):\n                    assert 0\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(failures=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_classname_instance.TestClass', name='test_method')

    @parametrize_families
    def test_classname_nested_dir(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        p = pytester.mkdir('sub').joinpath('test_hello.py')
        p.write_text('def test_func(): 0/0', encoding='utf-8')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(failures=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='sub.test_hello', name='test_func')

    @parametrize_families
    def test_internal_error(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('def pytest_runtest_protocol(): 0 / 0')
        pytester.makepyfile('def test_function(): pass')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(errors=1, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='pytest', name='internal')
        fnode = tnode.find_first_by_tag('error')
        fnode.assert_attr(message='internal error')
        assert 'Division' in fnode.toxml()

    @pytest.mark.parametrize('junit_logging', ['no', 'log', 'system-out', 'system-err', 'out-err', 'all'])
    @parametrize_families
    def test_failure_function(self, pytester: Pytester, junit_logging, run_and_parse: RunAndParse, xunit_family) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import logging\n            import sys\n\n            def test_fail():\n                print("hello-stdout")\n                sys.stderr.write("hello-stderr\\n")\n                logging.info(\'info msg\')\n                logging.warning(\'warning msg\')\n                raise ValueError(42)\n        ')
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging, family=xunit_family)
        assert result.ret, 'Expected ret > 0'
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(failures=1, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_failure_function', name='test_fail')
        fnode = tnode.find_first_by_tag('failure')
        fnode.assert_attr(message='ValueError: 42')
        assert 'ValueError' in fnode.toxml(), 'ValueError not included'
        if junit_logging in ['log', 'all']:
            logdata = tnode.find_first_by_tag('system-out')
            log_xml = logdata.toxml()
            assert logdata.tag == 'system-out', 'Expected tag: system-out'
            assert 'info msg' not in log_xml, 'Unexpected INFO message'
            assert 'warning msg' in log_xml, 'Missing WARN message'
        if junit_logging in ['system-out', 'out-err', 'all']:
            systemout = tnode.find_first_by_tag('system-out')
            systemout_xml = systemout.toxml()
            assert systemout.tag == 'system-out', 'Expected tag: system-out'
            assert 'info msg' not in systemout_xml, 'INFO message found in system-out'
            assert 'hello-stdout' in systemout_xml, "Missing 'hello-stdout' in system-out"
        if junit_logging in ['system-err', 'out-err', 'all']:
            systemerr = tnode.find_first_by_tag('system-err')
            systemerr_xml = systemerr.toxml()
            assert systemerr.tag == 'system-err', 'Expected tag: system-err'
            assert 'info msg' not in systemerr_xml, 'INFO message found in system-err'
            assert 'hello-stderr' in systemerr_xml, "Missing 'hello-stderr' in system-err"
            assert 'warning msg' not in systemerr_xml, 'WARN message found in system-err'
        if junit_logging == 'no':
            assert not tnode.find_by_tag('log'), 'Found unexpected content: log'
            assert not tnode.find_by_tag('system-out'), 'Found unexpected content: system-out'
            assert not tnode.find_by_tag('system-err'), 'Found unexpected content: system-err'

    @parametrize_families
    def test_failure_verbose_message(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import sys\n            def test_fail():\n                assert 0, "An error"\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        node = dom.find_first_by_tag('testsuite')
        tnode = node.find_first_by_tag('testcase')
        fnode = tnode.find_first_by_tag('failure')
        fnode.assert_attr(message='AssertionError: An error\nassert 0')

    @parametrize_families
    def test_failure_escape(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.parametrize(\'arg1\', "<&\'", ids="<&\'")\n            def test_func(arg1):\n                print(arg1)\n                assert 0\n        ')
        (result, dom) = run_and_parse('-o', 'junit_logging=system-out', family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(failures=3, tests=3)
        for (index, char) in enumerate("<&'"):
            tnode = node.find_nth_by_tag('testcase', index)
            tnode.assert_attr(classname='test_failure_escape', name='test_func[%s]' % char)
            sysout = tnode.find_first_by_tag('system-out')
            text = sysout.text
            assert '%s\n' % char in text

    @parametrize_families
    def test_junit_prefixing(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            def test_func():\n                assert 0\n            class TestHello(object):\n                def test_hello(self):\n                    pass\n        ')
        (result, dom) = run_and_parse('--junitprefix=xyz', family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(failures=1, tests=2)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='xyz.test_junit_prefixing', name='test_func')
        tnode = node.find_nth_by_tag('testcase', 1)
        tnode.assert_attr(classname='xyz.test_junit_prefixing.TestHello', name='test_hello')

    @parametrize_families
    def test_xfailure_function(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            def test_xfail():\n                pytest.xfail("42")\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert not result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=1, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_xfailure_function', name='test_xfail')
        fnode = tnode.find_first_by_tag('skipped')
        fnode.assert_attr(type='pytest.xfail', message='42')

    @parametrize_families
    def test_xfailure_marker(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.xfail(reason="42")\n            def test_xfail():\n                assert False\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        assert not result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=1, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_xfailure_marker', name='test_xfail')
        fnode = tnode.find_first_by_tag('skipped')
        fnode.assert_attr(type='pytest.xfail', message='42')

    @pytest.mark.parametrize('junit_logging', ['no', 'log', 'system-out', 'system-err', 'out-err', 'all'])
    def test_xfail_captures_output_once(self, pytester: Pytester, junit_logging: str, run_and_parse: RunAndParse) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile("\n            import sys\n            import pytest\n\n            @pytest.mark.xfail()\n            def test_fail():\n                sys.stdout.write('XFAIL This is stdout')\n                sys.stderr.write('XFAIL This is stderr')\n                assert 0\n        ")
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging)
        node = dom.find_first_by_tag('testsuite')
        tnode = node.find_first_by_tag('testcase')
        if junit_logging in ['system-err', 'out-err', 'all']:
            assert len(tnode.find_by_tag('system-err')) == 1
        else:
            assert len(tnode.find_by_tag('system-err')) == 0
        if junit_logging in ['log', 'system-out', 'out-err', 'all']:
            assert len(tnode.find_by_tag('system-out')) == 1
        else:
            assert len(tnode.find_by_tag('system-out')) == 0

    @parametrize_families
    def test_xfailure_xpass(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.xfail\n            def test_xpass():\n                pass\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=0, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_xfailure_xpass', name='test_xpass')

    @parametrize_families
    def test_xfailure_xpass_strict(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.xfail(strict=True, reason="This needs to fail!")\n            def test_xpass():\n                pass\n        ')
        (result, dom) = run_and_parse(family=xunit_family)
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(skipped=0, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(classname='test_xfailure_xpass_strict', name='test_xpass')
        fnode = tnode.find_first_by_tag('failure')
        fnode.assert_attr(message='[XPASS(strict)] This needs to fail!')

    @parametrize_families
    def test_collect_error(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            return 10
        pytester.makepyfile('syntax error')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(errors=1, tests=1)
        tnode = node.find_first_by_tag('testcase')
        fnode = tnode.find_first_by_tag('error')
        fnode.assert_attr(message='collection failure')
        assert 'SyntaxError' in fnode.toxml()

    def test_unicode(self, pytester: Pytester, run_and_parse: RunAndParse) -> None:
        if False:
            while True:
                i = 10
        value = 'hxÄ\x85Ä\x87\n'
        pytester.makepyfile('            # coding: latin1\n            def test_hello():\n                print(%r)\n                assert 0\n            ' % value)
        (result, dom) = run_and_parse()
        assert result.ret == 1
        tnode = dom.find_first_by_tag('testcase')
        fnode = tnode.find_first_by_tag('failure')
        assert 'hx' in fnode.toxml()

    def test_assertion_binchars(self, pytester: Pytester, run_and_parse: RunAndParse) -> None:
        if False:
            i = 10
            return i + 15
        "This test did fail when the escaping wasn't strict."
        pytester.makepyfile("\n\n            M1 = '\x01\x02\x03\x04'\n            M2 = '\x01\x02\x03\x05'\n\n            def test_str_compare():\n                assert M1 == M2\n            ")
        (result, dom) = run_and_parse()
        print(dom.toxml())

    @pytest.mark.parametrize('junit_logging', ['no', 'system-out'])
    def test_pass_captures_stdout(self, pytester: Pytester, run_and_parse: RunAndParse, junit_logging: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            def test_pass():\n                print('hello-stdout')\n        ")
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging)
        node = dom.find_first_by_tag('testsuite')
        pnode = node.find_first_by_tag('testcase')
        if junit_logging == 'no':
            assert not node.find_by_tag('system-out'), 'system-out should not be generated'
        if junit_logging == 'system-out':
            systemout = pnode.find_first_by_tag('system-out')
            assert 'hello-stdout' in systemout.toxml(), "'hello-stdout' should be in system-out"

    @pytest.mark.parametrize('junit_logging', ['no', 'system-err'])
    def test_pass_captures_stderr(self, pytester: Pytester, run_and_parse: RunAndParse, junit_logging: str) -> None:
        if False:
            return 10
        pytester.makepyfile("\n            import sys\n            def test_pass():\n                sys.stderr.write('hello-stderr')\n        ")
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging)
        node = dom.find_first_by_tag('testsuite')
        pnode = node.find_first_by_tag('testcase')
        if junit_logging == 'no':
            assert not node.find_by_tag('system-err'), 'system-err should not be generated'
        if junit_logging == 'system-err':
            systemerr = pnode.find_first_by_tag('system-err')
            assert 'hello-stderr' in systemerr.toxml(), "'hello-stderr' should be in system-err"

    @pytest.mark.parametrize('junit_logging', ['no', 'system-out'])
    def test_setup_error_captures_stdout(self, pytester: Pytester, run_and_parse: RunAndParse, junit_logging: str) -> None:
        if False:
            return 10
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture\n            def arg(request):\n                print('hello-stdout')\n                raise ValueError()\n            def test_function(arg):\n                pass\n        ")
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging)
        node = dom.find_first_by_tag('testsuite')
        pnode = node.find_first_by_tag('testcase')
        if junit_logging == 'no':
            assert not node.find_by_tag('system-out'), 'system-out should not be generated'
        if junit_logging == 'system-out':
            systemout = pnode.find_first_by_tag('system-out')
            assert 'hello-stdout' in systemout.toxml(), "'hello-stdout' should be in system-out"

    @pytest.mark.parametrize('junit_logging', ['no', 'system-err'])
    def test_setup_error_captures_stderr(self, pytester: Pytester, run_and_parse: RunAndParse, junit_logging: str) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            import sys\n            import pytest\n\n            @pytest.fixture\n            def arg(request):\n                sys.stderr.write('hello-stderr')\n                raise ValueError()\n            def test_function(arg):\n                pass\n        ")
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging)
        node = dom.find_first_by_tag('testsuite')
        pnode = node.find_first_by_tag('testcase')
        if junit_logging == 'no':
            assert not node.find_by_tag('system-err'), 'system-err should not be generated'
        if junit_logging == 'system-err':
            systemerr = pnode.find_first_by_tag('system-err')
            assert 'hello-stderr' in systemerr.toxml(), "'hello-stderr' should be in system-err"

    @pytest.mark.parametrize('junit_logging', ['no', 'system-out'])
    def test_avoid_double_stdout(self, pytester: Pytester, run_and_parse: RunAndParse, junit_logging: str) -> None:
        if False:
            return 10
        pytester.makepyfile("\n            import sys\n            import pytest\n\n            @pytest.fixture\n            def arg(request):\n                yield\n                sys.stdout.write('hello-stdout teardown')\n                raise ValueError()\n            def test_function(arg):\n                sys.stdout.write('hello-stdout call')\n        ")
        (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging)
        node = dom.find_first_by_tag('testsuite')
        pnode = node.find_first_by_tag('testcase')
        if junit_logging == 'no':
            assert not node.find_by_tag('system-out'), 'system-out should not be generated'
        if junit_logging == 'system-out':
            systemout = pnode.find_first_by_tag('system-out')
            assert 'hello-stdout call' in systemout.toxml()
            assert 'hello-stdout teardown' in systemout.toxml()

def test_mangle_test_address() -> None:
    if False:
        while True:
            i = 10
    from _pytest.junitxml import mangle_test_address
    address = '::'.join(['a/my.py.thing.py', 'Class', 'method', '[a-1-::]'])
    newnames = mangle_test_address(address)
    assert newnames == ['a.my.py.thing', 'Class', 'method', '[a-1-::]']

def test_dont_configure_on_workers(tmp_path: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    gotten: List[object] = []

    class FakeConfig:
        if TYPE_CHECKING:
            workerinput = None

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.pluginmanager = self
            self.option = self
            self.stash = Stash()

        def getini(self, name):
            if False:
                return 10
            return 'pytest'
        junitprefix = None
        xmlpath = str(tmp_path.joinpath('junix.xml'))
        register = gotten.append
    fake_config = cast(Config, FakeConfig())
    from _pytest import junitxml
    junitxml.pytest_configure(fake_config)
    assert len(gotten) == 1
    FakeConfig.workerinput = None
    junitxml.pytest_configure(fake_config)
    assert len(gotten) == 1

class TestNonPython:

    @parametrize_families
    def test_summing_simple(self, pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeconftest('\n            import pytest\n            def pytest_collect_file(file_path, parent):\n                if file_path.suffix == ".xyz":\n                    return MyItem.from_parent(name=file_path.name, parent=parent)\n            class MyItem(pytest.Item):\n                def runtest(self):\n                    raise ValueError(42)\n                def repr_failure(self, excinfo):\n                    return "custom item runtest failed"\n        ')
        pytester.path.joinpath('myfile.xyz').write_text('hello', encoding='utf-8')
        (result, dom) = run_and_parse(family=xunit_family)
        assert result.ret
        node = dom.find_first_by_tag('testsuite')
        node.assert_attr(errors=0, failures=1, skipped=0, tests=1)
        tnode = node.find_first_by_tag('testcase')
        tnode.assert_attr(name='myfile.xyz')
        fnode = tnode.find_first_by_tag('failure')
        fnode.assert_attr(message='custom item runtest failed')
        assert 'custom item runtest failed' in fnode.toxml()

@pytest.mark.parametrize('junit_logging', ['no', 'system-out'])
def test_nullbyte(pytester: Pytester, junit_logging: str) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile("\n        import sys\n        def test_print_nullbyte():\n            sys.stdout.write('Here the null -->' + chr(0) + '<--')\n            sys.stdout.write('In repr form -->' + repr(chr(0)) + '<--')\n            assert False\n    ")
    xmlf = pytester.path.joinpath('junit.xml')
    pytester.runpytest('--junitxml=%s' % xmlf, '-o', 'junit_logging=%s' % junit_logging)
    text = xmlf.read_text(encoding='utf-8')
    assert '\x00' not in text
    if junit_logging == 'system-out':
        assert '#x00' in text
    if junit_logging == 'no':
        assert '#x00' not in text

@pytest.mark.parametrize('junit_logging', ['no', 'system-out'])
def test_nullbyte_replace(pytester: Pytester, junit_logging: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile("\n        import sys\n        def test_print_nullbyte():\n            sys.stdout.write('Here the null -->' + chr(0) + '<--')\n            sys.stdout.write('In repr form -->' + repr(chr(0)) + '<--')\n            assert False\n    ")
    xmlf = pytester.path.joinpath('junit.xml')
    pytester.runpytest('--junitxml=%s' % xmlf, '-o', 'junit_logging=%s' % junit_logging)
    text = xmlf.read_text(encoding='utf-8')
    if junit_logging == 'system-out':
        assert '#x0' in text
    if junit_logging == 'no':
        assert '#x0' not in text

def test_invalid_xml_escape() -> None:
    if False:
        return 10
    invalid = (0, 1, 11, 12, 14, 25, 27, 55296, 57343, 65534, 65535)
    valid = (9, 10, 32)
    for i in invalid:
        got = bin_xml_escape(chr(i))
        if i <= 255:
            expected = '#x%02X' % i
        else:
            expected = '#x%04X' % i
        assert got == expected
    for i in valid:
        assert chr(i) == bin_xml_escape(chr(i))

def test_logxml_path_expansion(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    home_tilde = Path(os.path.expanduser('~')).joinpath('test.xml')
    xml_tilde = LogXML(Path('~', 'test.xml'), None)
    assert xml_tilde.logfile == str(home_tilde)
    monkeypatch.setenv('HOME', str(tmp_path))
    home_var = os.path.normpath(os.path.expandvars('$HOME/test.xml'))
    xml_var = LogXML(Path('$HOME', 'test.xml'), None)
    assert xml_var.logfile == str(home_var)

def test_logxml_changingdir(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        def test_func():\n            import os\n            os.chdir("a")\n    ')
    pytester.mkdir('a')
    result = pytester.runpytest('--junitxml=a/x.xml')
    assert result.ret == 0
    assert pytester.path.joinpath('a/x.xml').exists()

def test_logxml_makedir(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    '--junitxml should automatically create directories for the xml file'
    pytester.makepyfile('\n        def test_pass():\n            pass\n    ')
    result = pytester.runpytest('--junitxml=path/to/results.xml')
    assert result.ret == 0
    assert pytester.path.joinpath('path/to/results.xml').exists()

def test_logxml_check_isdir(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    'Give an error if --junit-xml is a directory (#2089)'
    result = pytester.runpytest('--junit-xml=.')
    result.stderr.fnmatch_lines(['*--junitxml must be a filename*'])

def test_escaped_parametrized_names_xml(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('        import pytest\n        @pytest.mark.parametrize(\'char\', ["\\x00"])\n        def test_func(char):\n            assert char\n        ')
    (result, dom) = run_and_parse()
    assert result.ret == 0
    node = dom.find_first_by_tag('testcase')
    node.assert_attr(name='test_func[\\x00]')

def test_double_colon_split_function_issue469(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        import pytest\n        @pytest.mark.parametrize(\'param\', ["double::colon"])\n        def test_func(param):\n            pass\n    ')
    (result, dom) = run_and_parse()
    assert result.ret == 0
    node = dom.find_first_by_tag('testcase')
    node.assert_attr(classname='test_double_colon_split_function_issue469')
    node.assert_attr(name='test_func[double::colon]')

def test_double_colon_split_method_issue469(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        import pytest\n        class TestClass(object):\n            @pytest.mark.parametrize(\'param\', ["double::colon"])\n            def test_func(self, param):\n                pass\n    ')
    (result, dom) = run_and_parse()
    assert result.ret == 0
    node = dom.find_first_by_tag('testcase')
    node.assert_attr(classname='test_double_colon_split_method_issue469.TestClass')
    node.assert_attr(name='test_func[double::colon]')

def test_unicode_issue368(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    path = pytester.path.joinpath('test.xml')
    log = LogXML(str(path), None)
    ustr = 'ВНИ!'

    class Report(BaseReport):
        longrepr = ustr
        sections: List[Tuple[str, str]] = []
        nodeid = 'something'
        location = ('tests/filename.py', 42, 'TestClass.method')
        when = 'teardown'
    test_report = cast(TestReport, Report())
    log.pytest_sessionstart()
    node_reporter = log._opentestcase(test_report)
    node_reporter.append_failure(test_report)
    node_reporter.append_collect_error(test_report)
    node_reporter.append_collect_skipped(test_report)
    node_reporter.append_error(test_report)
    test_report.longrepr = ('filename', 1, ustr)
    node_reporter.append_skipped(test_report)
    test_report.longrepr = ('filename', 1, 'Skipped: 卡嘣嘣')
    node_reporter.append_skipped(test_report)
    test_report.wasxfail = ustr
    node_reporter.append_skipped(test_report)
    log.pytest_sessionfinish()

def test_record_property(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture\n        def other(record_property):\n            record_property("bar", 1)\n        def test_record(record_property, other):\n            record_property("foo", "<1");\n    ')
    (result, dom) = run_and_parse()
    node = dom.find_first_by_tag('testsuite')
    tnode = node.find_first_by_tag('testcase')
    psnode = tnode.find_first_by_tag('properties')
    pnodes = psnode.find_by_tag('property')
    pnodes[0].assert_attr(name='bar', value='1')
    pnodes[1].assert_attr(name='foo', value='<1')
    result.stdout.fnmatch_lines(['*= 1 passed in *'])

def test_record_property_on_test_and_teardown_failure(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture\n        def other(record_property):\n            record_property("bar", 1)\n            yield\n            assert 0\n\n        def test_record(record_property, other):\n            record_property("foo", "<1")\n            assert 0\n    ')
    (result, dom) = run_and_parse()
    node = dom.find_first_by_tag('testsuite')
    tnodes = node.find_by_tag('testcase')
    for tnode in tnodes:
        psnode = tnode.find_first_by_tag('properties')
        assert psnode, f"testcase didn't had expected properties:\n{tnode}"
        pnodes = psnode.find_by_tag('property')
        pnodes[0].assert_attr(name='bar', value='1')
        pnodes[1].assert_attr(name='foo', value='<1')
    result.stdout.fnmatch_lines(['*= 1 failed, 1 error *'])

def test_record_property_same_name(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        def test_record_with_same_name(record_property):\n            record_property("foo", "bar")\n            record_property("foo", "baz")\n    ')
    (result, dom) = run_and_parse()
    node = dom.find_first_by_tag('testsuite')
    tnode = node.find_first_by_tag('testcase')
    psnode = tnode.find_first_by_tag('properties')
    pnodes = psnode.find_by_tag('property')
    pnodes[0].assert_attr(name='foo', value='bar')
    pnodes[1].assert_attr(name='foo', value='baz')

@pytest.mark.parametrize('fixture_name', ['record_property', 'record_xml_attribute'])
def test_record_fixtures_without_junitxml(pytester: Pytester, fixture_name: str) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        def test_record({fixture_name}):\n            {fixture_name}("foo", "bar")\n    '.format(fixture_name=fixture_name))
    result = pytester.runpytest()
    assert result.ret == 0

@pytest.mark.filterwarnings('default')
def test_record_attribute(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        print('Hello World!')
    pytester.makeini('\n        [pytest]\n        junit_family = xunit1\n    ')
    pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture\n        def other(record_xml_attribute):\n            record_xml_attribute("bar", 1)\n        def test_record(record_xml_attribute, other):\n            record_xml_attribute("foo", "<1");\n    ')
    (result, dom) = run_and_parse()
    node = dom.find_first_by_tag('testsuite')
    tnode = node.find_first_by_tag('testcase')
    tnode.assert_attr(bar='1')
    tnode.assert_attr(foo='<1')
    result.stdout.fnmatch_lines(['*test_record_attribute.py:6:*record_xml_attribute is an experimental feature'])

@pytest.mark.filterwarnings('default')
@pytest.mark.parametrize('fixture_name', ['record_xml_attribute', 'record_property'])
def test_record_fixtures_xunit2(pytester: Pytester, fixture_name: str, run_and_parse: RunAndParse) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Ensure record_xml_attribute and record_property drop values when outside of legacy family.'
    pytester.makeini('\n        [pytest]\n        junit_family = xunit2\n    ')
    pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture\n        def other({fixture_name}):\n            {fixture_name}("bar", 1)\n        def test_record({fixture_name}, other):\n            {fixture_name}("foo", "<1");\n    '.format(fixture_name=fixture_name))
    (result, dom) = run_and_parse(family=None)
    expected_lines = []
    if fixture_name == 'record_xml_attribute':
        expected_lines.append('*test_record_fixtures_xunit2.py:6:*record_xml_attribute is an experimental feature')
    expected_lines = ["*test_record_fixtures_xunit2.py:6:*{fixture_name} is incompatible with junit_family 'xunit2' (use 'legacy' or 'xunit1')".format(fixture_name=fixture_name)]
    result.stdout.fnmatch_lines(expected_lines)

def test_random_report_log_xdist(pytester: Pytester, monkeypatch: MonkeyPatch, run_and_parse: RunAndParse) -> None:
    if False:
        for i in range(10):
            print('nop')
    '`xdist` calls pytest_runtest_logreport as they are executed by the workers,\n    with nodes from several nodes overlapping, so junitxml must cope with that\n    to produce correct reports (#1064).'
    pytest.importorskip('xdist')
    monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', raising=False)
    pytester.makepyfile("\n        import pytest, time\n        @pytest.mark.parametrize('i', list(range(30)))\n        def test_x(i):\n            assert i != 22\n    ")
    (_, dom) = run_and_parse('-n2')
    suite_node = dom.find_first_by_tag('testsuite')
    failed = []
    for case_node in suite_node.find_by_tag('testcase'):
        if case_node.find_first_by_tag('failure'):
            failed.append(case_node['name'])
    assert failed == ['test_x[22]']

@parametrize_families
def test_root_testsuites_tag(pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        def test_x():\n            pass\n    ')
    (_, dom) = run_and_parse(family=xunit_family)
    root = dom.get_unique_child
    assert root.tag == 'testsuites'
    suite_node = root.get_unique_child
    assert suite_node.tag == 'testsuite'

def test_runs_twice(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        i = 10
        return i + 15
    f = pytester.makepyfile('\n        def test_pass():\n            pass\n    ')
    (result, dom) = run_and_parse(f, f)
    result.stdout.no_fnmatch_line('*INTERNALERROR*')
    (first, second) = (x['classname'] for x in dom.find_by_tag('testcase'))
    assert first == second

def test_runs_twice_xdist(pytester: Pytester, monkeypatch: MonkeyPatch, run_and_parse: RunAndParse) -> None:
    if False:
        i = 10
        return i + 15
    pytest.importorskip('xdist')
    monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD')
    f = pytester.makepyfile('\n        def test_pass():\n            pass\n    ')
    (result, dom) = run_and_parse(f, '--dist', 'each', '--tx', '2*popen')
    result.stdout.no_fnmatch_line('*INTERNALERROR*')
    (first, second) = (x['classname'] for x in dom.find_by_tag('testcase'))
    assert first == second

def test_fancy_items_regression(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        while True:
            i = 10
    pytester.makeconftest("\n        import pytest\n        class FunItem(pytest.Item):\n            def runtest(self):\n                pass\n        class NoFunItem(pytest.Item):\n            def runtest(self):\n                pass\n\n        class FunCollector(pytest.File):\n            def collect(self):\n                return [\n                    FunItem.from_parent(name='a', parent=self),\n                    NoFunItem.from_parent(name='a', parent=self),\n                    NoFunItem.from_parent(name='b', parent=self),\n                ]\n\n        def pytest_collect_file(file_path, parent):\n            if file_path.suffix == '.py':\n                return FunCollector.from_parent(path=file_path, parent=parent)\n    ")
    pytester.makepyfile('\n        def test_pass():\n            pass\n    ')
    (result, dom) = run_and_parse()
    result.stdout.no_fnmatch_line('*INTERNALERROR*')
    items = sorted(('%(classname)s %(name)s' % x for x in dom.find_by_tag('testcase')))
    import pprint
    pprint.pprint(items)
    assert items == ['conftest a', 'conftest a', 'conftest b', 'test_fancy_items_regression a', 'test_fancy_items_regression a', 'test_fancy_items_regression b', 'test_fancy_items_regression test_pass']

@parametrize_families
def test_global_properties(pytester: Pytester, xunit_family: str) -> None:
    if False:
        while True:
            i = 10
    path = pytester.path.joinpath('test_global_properties.xml')
    log = LogXML(str(path), None, family=xunit_family)

    class Report(BaseReport):
        sections: List[Tuple[str, str]] = []
        nodeid = 'test_node_id'
    log.pytest_sessionstart()
    log.add_global_property('foo', '1')
    log.add_global_property('bar', '2')
    log.pytest_sessionfinish()
    dom = minidom.parse(str(path))
    properties = dom.getElementsByTagName('properties')
    assert properties.length == 1, 'There must be one <properties> node'
    property_list = dom.getElementsByTagName('property')
    assert property_list.length == 2, 'There most be only 2 property nodes'
    expected = {'foo': '1', 'bar': '2'}
    actual = {}
    for p in property_list:
        k = str(p.getAttribute('name'))
        v = str(p.getAttribute('value'))
        actual[k] = v
    assert actual == expected

def test_url_property(pytester: Pytester) -> None:
    if False:
        return 10
    test_url = 'http://www.github.com/pytest-dev'
    path = pytester.path.joinpath('test_url_property.xml')
    log = LogXML(str(path), None)

    class Report(BaseReport):
        longrepr = 'FooBarBaz'
        sections: List[Tuple[str, str]] = []
        nodeid = 'something'
        location = ('tests/filename.py', 42, 'TestClass.method')
        url = test_url
    test_report = cast(TestReport, Report())
    log.pytest_sessionstart()
    node_reporter = log._opentestcase(test_report)
    node_reporter.append_failure(test_report)
    log.pytest_sessionfinish()
    test_case = minidom.parse(str(path)).getElementsByTagName('testcase')[0]
    assert test_case.getAttribute('url') == test_url, 'The URL did not get written to the xml'

@parametrize_families
def test_record_testsuite_property(pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile('\n        def test_func1(record_testsuite_property):\n            record_testsuite_property("stats", "all good")\n\n        def test_func2(record_testsuite_property):\n            record_testsuite_property("stats", 10)\n    ')
    (result, dom) = run_and_parse(family=xunit_family)
    assert result.ret == 0
    node = dom.find_first_by_tag('testsuite')
    properties_node = node.find_first_by_tag('properties')
    p1_node = properties_node.find_nth_by_tag('property', 0)
    p2_node = properties_node.find_nth_by_tag('property', 1)
    p1_node.assert_attr(name='stats', value='all good')
    p2_node.assert_attr(name='stats', value='10')

def test_record_testsuite_property_junit_disabled(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        def test_func1(record_testsuite_property):\n            record_testsuite_property("stats", "all good")\n    ')
    result = pytester.runpytest()
    assert result.ret == 0

@pytest.mark.parametrize('junit', [True, False])
def test_record_testsuite_property_type_checking(pytester: Pytester, junit: bool) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def test_func1(record_testsuite_property):\n            record_testsuite_property(1, 2)\n    ')
    args = ('--junitxml=tests.xml',) if junit else ()
    result = pytester.runpytest(*args)
    assert result.ret == 1
    result.stdout.fnmatch_lines(['*TypeError: name parameter needs to be a string, but int given'])

@pytest.mark.parametrize('suite_name', ['my_suite', ''])
@parametrize_families
def test_set_suite_name(pytester: Pytester, suite_name: str, run_and_parse: RunAndParse, xunit_family: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    if suite_name:
        pytester.makeini('\n            [pytest]\n            junit_suite_name={suite_name}\n            junit_family={family}\n        '.format(suite_name=suite_name, family=xunit_family))
        expected = suite_name
    else:
        expected = 'pytest'
    pytester.makepyfile('\n        import pytest\n\n        def test_func():\n            pass\n    ')
    (result, dom) = run_and_parse(family=xunit_family)
    assert result.ret == 0
    node = dom.find_first_by_tag('testsuite')
    node.assert_attr(name=expected)

def test_escaped_skipreason_issue3533(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile("\n        import pytest\n        @pytest.mark.skip(reason='1 <> 2')\n        def test_skip():\n            pass\n    ")
    (_, dom) = run_and_parse()
    node = dom.find_first_by_tag('testcase')
    snode = node.find_first_by_tag('skipped')
    assert '1 <> 2' in snode.text
    snode.assert_attr(message='1 <> 2')

def test_escaped_setup_teardown_error(pytester: Pytester, run_and_parse: RunAndParse) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture()\n        def my_setup():\n            raise Exception("error: \x1b[31mred\x1b[m")\n\n        def test_esc(my_setup):\n            pass\n    ')
    (_, dom) = run_and_parse()
    node = dom.find_first_by_tag('testcase')
    snode = node.find_first_by_tag('error')
    assert '#x1B[31mred#x1B[m' in snode['message']
    assert '#x1B[31mred#x1B[m' in snode.text

@parametrize_families
def test_logging_passing_tests_disabled_does_not_log_test_output(pytester: Pytester, run_and_parse: RunAndParse, xunit_family: str) -> None:
    if False:
        while True:
            i = 10
    pytester.makeini('\n        [pytest]\n        junit_log_passing_tests=False\n        junit_logging=system-out\n        junit_family={family}\n    '.format(family=xunit_family))
    pytester.makepyfile("\n        import pytest\n        import logging\n        import sys\n\n        def test_func():\n            sys.stdout.write('This is stdout')\n            sys.stderr.write('This is stderr')\n            logging.warning('hello')\n    ")
    (result, dom) = run_and_parse(family=xunit_family)
    assert result.ret == 0
    node = dom.find_first_by_tag('testcase')
    assert len(node.find_by_tag('system-err')) == 0
    assert len(node.find_by_tag('system-out')) == 0

@parametrize_families
@pytest.mark.parametrize('junit_logging', ['no', 'system-out', 'system-err'])
def test_logging_passing_tests_disabled_logs_output_for_failing_test_issue5430(pytester: Pytester, junit_logging: str, run_and_parse: RunAndParse, xunit_family: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makeini('\n        [pytest]\n        junit_log_passing_tests=False\n        junit_family={family}\n    '.format(family=xunit_family))
    pytester.makepyfile("\n        import pytest\n        import logging\n        import sys\n\n        def test_func():\n            logging.warning('hello')\n            assert 0\n    ")
    (result, dom) = run_and_parse('-o', 'junit_logging=%s' % junit_logging, family=xunit_family)
    assert result.ret == 1
    node = dom.find_first_by_tag('testcase')
    if junit_logging == 'system-out':
        assert len(node.find_by_tag('system-err')) == 0
        assert len(node.find_by_tag('system-out')) == 1
    elif junit_logging == 'system-err':
        assert len(node.find_by_tag('system-err')) == 1
        assert len(node.find_by_tag('system-out')) == 0
    else:
        assert junit_logging == 'no'
        assert len(node.find_by_tag('system-err')) == 0
        assert len(node.find_by_tag('system-out')) == 0