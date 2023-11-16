"""Terminal reporting of the full testing process."""
import collections
import os
import sys
import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from typing import Dict
from typing import List
from typing import Tuple
import pluggy
import _pytest.config
import _pytest.terminal
import pytest
from _pytest._io.wcwidth import wcswidth
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.reports import BaseReport
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.terminal import _folded_skips
from _pytest.terminal import _format_trimmed
from _pytest.terminal import _get_line_with_reprcrash_message
from _pytest.terminal import _get_raw_skip_reason
from _pytest.terminal import _plugin_nameversions
from _pytest.terminal import getreportopt
from _pytest.terminal import TerminalReporter
DistInfo = collections.namedtuple('DistInfo', ['project_name', 'version'])
TRANS_FNMATCH = str.maketrans({'[': '[[]', ']': '[]]'})

class Option:

    def __init__(self, verbosity=0):
        if False:
            while True:
                i = 10
        self.verbosity = verbosity

    @property
    def args(self):
        if False:
            print('Hello World!')
        values = []
        values.append('--verbosity=%d' % self.verbosity)
        return values

@pytest.fixture(params=[Option(verbosity=0), Option(verbosity=1), Option(verbosity=-1)], ids=['default', 'verbose', 'quiet'])
def option(request):
    if False:
        for i in range(10):
            print('nop')
    return request.param

@pytest.mark.parametrize('input,expected', [([DistInfo(project_name='test', version=1)], ['test-1']), ([DistInfo(project_name='pytest-test', version=1)], ['test-1']), ([DistInfo(project_name='test', version=1), DistInfo(project_name='test', version=1)], ['test-1'])], ids=['normal', 'prefix-strip', 'deduplicate'])
def test_plugin_nameversion(input, expected):
    if False:
        while True:
            i = 10
    pluginlist = [(None, x) for x in input]
    result = _plugin_nameversions(pluginlist)
    assert result == expected

class TestTerminal:

    def test_pass_skip_fail(self, pytester: Pytester, option) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            def test_ok():\n                pass\n            def test_skip():\n                pytest.skip("xx")\n            def test_func():\n                assert 0\n        ')
        result = pytester.runpytest(*option.args)
        if option.verbosity > 0:
            result.stdout.fnmatch_lines(['*test_pass_skip_fail.py::test_ok PASS*', '*test_pass_skip_fail.py::test_skip SKIP*', '*test_pass_skip_fail.py::test_func FAIL*'])
        elif option.verbosity == 0:
            result.stdout.fnmatch_lines(['*test_pass_skip_fail.py .sF*'])
        else:
            result.stdout.fnmatch_lines(['.sF*'])
        result.stdout.fnmatch_lines(['    def test_func():', '>       assert 0', 'E       assert 0'])

    def test_internalerror(self, pytester: Pytester, linecomp) -> None:
        if False:
            return 10
        modcol = pytester.getmodulecol('def test_one(): pass')
        rep = TerminalReporter(modcol.config, file=linecomp.stringio)
        with pytest.raises(ValueError) as excinfo:
            raise ValueError('hello')
        rep.pytest_internalerror(excinfo.getrepr())
        linecomp.assert_contains_lines(['INTERNALERROR> *ValueError*hello*'])

    def test_writeline(self, pytester: Pytester, linecomp) -> None:
        if False:
            while True:
                i = 10
        modcol = pytester.getmodulecol('def test_one(): pass')
        rep = TerminalReporter(modcol.config, file=linecomp.stringio)
        rep.write_fspath_result(modcol.nodeid, '.')
        rep.write_line('hello world')
        lines = linecomp.stringio.getvalue().split('\n')
        assert not lines[0]
        assert lines[1].endswith(modcol.name + ' .')
        assert lines[2] == 'hello world'

    def test_show_runtest_logstart(self, pytester: Pytester, linecomp) -> None:
        if False:
            while True:
                i = 10
        item = pytester.getitem('def test_func(): pass')
        tr = TerminalReporter(item.config, file=linecomp.stringio)
        item.config.pluginmanager.register(tr)
        location = item.reportinfo()
        tr.config.hook.pytest_runtest_logstart(nodeid=item.nodeid, location=location, fspath=str(item.path))
        linecomp.assert_contains_lines(['*test_show_runtest_logstart.py*'])

    def test_runtest_location_shown_before_test_starts(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            def test_1():\n                import time\n                time.sleep(20)\n        ')
        child = pytester.spawn_pytest('')
        child.expect('.*test_runtest_location.*py')
        child.sendeof()
        child.kill(15)

    def test_report_collect_after_half_a_second(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            return 10
        'Test for "collecting" being updated after 0.5s'
        pytester.makepyfile(**{'test1.py': '\n                import _pytest.terminal\n\n                _pytest.terminal.REPORT_COLLECTING_RESOLUTION = 0\n\n                def test_1():\n                    pass\n                    ', 'test2.py': 'def test_2(): pass'})
        monkeypatch.setenv('PY_COLORS', '1')
        child = pytester.spawn_pytest('-v test1.py test2.py')
        child.expect('collecting \\.\\.\\.')
        child.expect('collecting 1 item')
        child.expect('collecting 2 items')
        child.expect('collected 2 items')
        rest = child.read().decode('utf8')
        assert '= \x1b[32m\x1b[1m2 passed\x1b[0m\x1b[32m in' in rest

    def test_itemreport_subclasses_show_subclassed_file(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(**{'tests/test_p1': '\n            class BaseTests(object):\n                fail = False\n\n                def test_p1(self):\n                    if self.fail: assert 0\n                ', 'tests/test_p2': '\n            from test_p1 import BaseTests\n\n            class TestMore(BaseTests): pass\n                ', 'tests/test_p3.py': '\n            from test_p1 import BaseTests\n\n            BaseTests.fail = True\n\n            class TestMore(BaseTests): pass\n        '})
        result = pytester.runpytest('tests/test_p2.py', '--rootdir=tests')
        result.stdout.fnmatch_lines(['tests/test_p2.py .*', '=* 1 passed in *'])
        result = pytester.runpytest('-vv', '-rA', 'tests/test_p2.py', '--rootdir=tests')
        result.stdout.fnmatch_lines(['tests/test_p2.py::TestMore::test_p1 <- test_p1.py PASSED *', '*= short test summary info =*', 'PASSED tests/test_p2.py::TestMore::test_p1'])
        result = pytester.runpytest('-vv', '-rA', 'tests/test_p3.py', '--rootdir=tests')
        result.stdout.fnmatch_lines(['tests/test_p3.py::TestMore::test_p1 <- test_p1.py FAILED *', '*_ TestMore.test_p1 _*', '    def test_p1(self):', '>       if self.fail: assert 0', 'E       assert 0', '', 'tests/test_p1.py:5: AssertionError', '*= short test summary info =*', 'FAILED tests/test_p3.py::TestMore::test_p1 - assert 0', '*= 1 failed in *'])

    def test_itemreport_directclasses_not_shown_as_subclasses(self, pytester: Pytester) -> None:
        if False:
            return 10
        a = pytester.mkpydir('a123')
        a.joinpath('test_hello123.py').write_text(textwrap.dedent('                class TestClass(object):\n                    def test_method(self):\n                        pass\n                '), encoding='utf-8')
        result = pytester.runpytest('-vv')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*a123/test_hello123.py*PASS*'])
        result.stdout.no_fnmatch_line('* <- *')

    @pytest.mark.parametrize('fulltrace', ('', '--fulltrace'))
    def test_keyboard_interrupt(self, pytester: Pytester, fulltrace) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            def test_foobar():\n                assert 0\n            def test_spamegg():\n                import py; pytest.skip('skip me please!')\n            def test_interrupt_me():\n                raise KeyboardInterrupt   # simulating the user\n        ")
        result = pytester.runpytest(fulltrace, no_reraise_ctrlc=True)
        result.stdout.fnmatch_lines(['    def test_foobar():', '>       assert 0', 'E       assert 0', '*_keyboard_interrupt.py:6: KeyboardInterrupt*'])
        if fulltrace:
            result.stdout.fnmatch_lines(['*raise KeyboardInterrupt   # simulating the user*'])
        else:
            result.stdout.fnmatch_lines(['(to show a full traceback on KeyboardInterrupt use --full-trace)'])
        result.stdout.fnmatch_lines(['*KeyboardInterrupt*'])

    def test_keyboard_in_sessionstart(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeconftest('\n            def pytest_sessionstart():\n                raise KeyboardInterrupt\n        ')
        pytester.makepyfile('\n            def test_foobar():\n                pass\n        ')
        result = pytester.runpytest(no_reraise_ctrlc=True)
        assert result.ret == 2
        result.stdout.fnmatch_lines(['*KeyboardInterrupt*'])

    def test_collect_single_item(self, pytester: Pytester) -> None:
        if False:
            return 10
        "Use singular 'item' when reporting a single test item"
        pytester.makepyfile('\n            def test_foobar():\n                pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 1 item'])

    def test_rewrite(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        config = pytester.parseconfig()
        f = StringIO()
        monkeypatch.setattr(f, 'isatty', lambda *args: True)
        tr = TerminalReporter(config, f)
        tr._tw.fullwidth = 10
        tr.write('hello')
        tr.rewrite('hey', erase=True)
        assert f.getvalue() == 'hello' + '\r' + 'hey' + 6 * ' '

    def test_report_teststatus_explicit_markup(self, monkeypatch: MonkeyPatch, pytester: Pytester, color_mapping) -> None:
        if False:
            print('Hello World!')
        'Test that TerminalReporter handles markup explicitly provided by\n        a pytest_report_teststatus hook.'
        monkeypatch.setenv('PY_COLORS', '1')
        pytester.makeconftest("\n            def pytest_report_teststatus(report):\n                return 'foo', 'F', ('FOO', {'red': True})\n        ")
        pytester.makepyfile('\n            def test_foobar():\n                pass\n        ')
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['*{red}FOO{reset}*']))

    def test_verbose_skip_reason(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.skip(reason="123")\n            def test_1():\n                pass\n\n            @pytest.mark.xfail(reason="456")\n            def test_2():\n                pass\n\n            @pytest.mark.xfail(reason="789")\n            def test_3():\n                assert False\n\n            @pytest.mark.xfail(reason="")\n            def test_4():\n                assert False\n\n            @pytest.mark.skip\n            def test_5():\n                pass\n\n            @pytest.mark.xfail\n            def test_6():\n                pass\n\n            def test_7():\n                pytest.skip()\n\n            def test_8():\n                pytest.skip("888 is great")\n\n            def test_9():\n                pytest.xfail()\n\n            def test_10():\n                pytest.xfail("It\'s 🕙 o\'clock")\n\n            @pytest.mark.skip(\n                reason="1 cannot do foobar because baz is missing due to I don\'t know what"\n            )\n            def test_long_skip():\n                pass\n\n            @pytest.mark.xfail(\n                reason="2 cannot do foobar because baz is missing due to I don\'t know what"\n            )\n            def test_long_xfail():\n                print(1 / 0)\n        ')
        common_output = ['test_verbose_skip_reason.py::test_1 SKIPPED (123) *', 'test_verbose_skip_reason.py::test_2 XPASS (456) *', 'test_verbose_skip_reason.py::test_3 XFAIL (789) *', 'test_verbose_skip_reason.py::test_4 XFAIL  *', 'test_verbose_skip_reason.py::test_5 SKIPPED (unconditional skip) *', 'test_verbose_skip_reason.py::test_6 XPASS  *', 'test_verbose_skip_reason.py::test_7 SKIPPED  *', 'test_verbose_skip_reason.py::test_8 SKIPPED (888 is great) *', 'test_verbose_skip_reason.py::test_9 XFAIL  *', "test_verbose_skip_reason.py::test_10 XFAIL (It's 🕙 o'clock) *"]
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(common_output + ['test_verbose_skip_reason.py::test_long_skip SKIPPED (1 cannot *...) *', 'test_verbose_skip_reason.py::test_long_xfail XFAIL (2 cannot *...) *'])
        result = pytester.runpytest('-vv')
        result.stdout.fnmatch_lines(common_output + ['test_verbose_skip_reason.py::test_long_skip SKIPPED (1 cannot do foobar', "because baz is missing due to I don't know what) *", 'test_verbose_skip_reason.py::test_long_xfail XFAIL (2 cannot do foobar', "because baz is missing due to I don't know what) *"])

class TestCollectonly:

    def test_collectonly_basic(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_func():\n                pass\n        ')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['<Module test_collectonly_basic.py>', '  <Function test_func>'])

    def test_collectonly_skipped_module(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            pytest.skip("hello")\n        ')
        result = pytester.runpytest('--collect-only', '-rs')
        result.stdout.fnmatch_lines(['*ERROR collecting*'])

    def test_collectonly_displays_test_description(self, pytester: Pytester, dummy_yaml_custom_test) -> None:
        if False:
            while True:
                i = 10
        'Used dummy_yaml_custom_test for an Item without ``obj``.'
        pytester.makepyfile("\n            def test_with_description():\n                '''  This test has a description.\n\n                  more1.\n                    more2.'''\n            ")
        result = pytester.runpytest('--collect-only', '--verbose')
        result.stdout.fnmatch_lines(['<YamlFile test1.yaml>', '  <YamlItem test1.yaml>', '<Module test_collectonly_displays_test_description.py>', '  <Function test_with_description>', '    This test has a description.', '    ', '    more1.', '      more2.'], consecutive=True)

    def test_collectonly_failed_module(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('raise ValueError(0)')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['*raise ValueError*', '*1 error*'])

    def test_collectonly_fatal(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makeconftest('\n            def pytest_collectstart(collector):\n                assert 0, "urgs"\n        ')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['*INTERNAL*args*'])
        assert result.ret == 3

    def test_collectonly_simple(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n            def test_func1():\n                pass\n            class TestClass(object):\n                def test_method(self):\n                    pass\n        ')
        result = pytester.runpytest('--collect-only', p)
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*<Module *.py>', '* <Function test_func1>', '* <Class TestClass>', '*   <Function test_method>'])

    def test_collectonly_error(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('import Errlkjqweqwe')
        result = pytester.runpytest('--collect-only', p)
        assert result.ret == 2
        result.stdout.fnmatch_lines(textwrap.dedent('                *ERROR*\n                *ImportError*\n                *No module named *Errlk*\n                *1 error*\n                ').strip())

    def test_collectonly_missing_path(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Issue 115: failure in parseargs will cause session not to\n        have the items attribute.'
        result = pytester.runpytest('--collect-only', 'uhm_missing_path')
        assert result.ret == 4
        result.stderr.fnmatch_lines(['*ERROR: file or directory not found: uhm_missing_path'])

    def test_collectonly_quiet(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('def test_foo(): pass')
        result = pytester.runpytest('--collect-only', '-q')
        result.stdout.fnmatch_lines(['*test_foo*'])

    def test_collectonly_more_quiet(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(test_fun='def test_foo(): pass')
        result = pytester.runpytest('--collect-only', '-qq')
        result.stdout.fnmatch_lines(['*test_fun.py: 1*'])

    def test_collect_only_summary_status(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Custom status depending on test selection using -k or -m. #7701.'
        pytester.makepyfile(test_collect_foo='\n            def test_foo(): pass\n            ', test_collect_bar='\n            def test_foobar(): pass\n            def test_bar(): pass\n            ')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines('*== 3 tests collected in * ==*')
        result = pytester.runpytest('--collect-only', 'test_collect_foo.py')
        result.stdout.fnmatch_lines('*== 1 test collected in * ==*')
        result = pytester.runpytest('--collect-only', '-k', 'foo')
        result.stdout.fnmatch_lines('*== 2/3 tests collected (1 deselected) in * ==*')
        result = pytester.runpytest('--collect-only', '-k', 'test_bar')
        result.stdout.fnmatch_lines('*== 1/3 tests collected (2 deselected) in * ==*')
        result = pytester.runpytest('--collect-only', '-k', 'invalid')
        result.stdout.fnmatch_lines('*== no tests collected (3 deselected) in * ==*')
        pytester.mkdir('no_tests_here')
        result = pytester.runpytest('--collect-only', 'no_tests_here')
        result.stdout.fnmatch_lines('*== no tests collected in * ==*')
        pytester.makepyfile(test_contains_error='\n            raise RuntimeError\n            ')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines('*== 3 tests collected, 1 error in * ==*')
        result = pytester.runpytest('--collect-only', '-k', 'foo')
        result.stdout.fnmatch_lines('*== 2/3 tests collected (1 deselected), 1 error in * ==*')

class TestFixtureReporting:

    def test_setup_fixture_error(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def setup_function(function):\n                print("setup func")\n                assert 0\n            def test_nada():\n                pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*ERROR at setup of test_nada*', '*setup_function(function):*', '*setup func*', '*assert 0*', '*1 error*'])
        assert result.ret != 0

    def test_teardown_fixture_error(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_nada():\n                pass\n            def teardown_function(function):\n                print("teardown func")\n                assert 0\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*ERROR at teardown*', '*teardown_function(function):*', '*assert 0*', '*Captured stdout*', '*teardown func*', '*1 passed*1 error*'])

    def test_teardown_fixture_error_and_test_failure(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_fail():\n                assert 0, "failingfunc"\n\n            def teardown_function(function):\n                print("teardown func")\n                assert False\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*ERROR at teardown of test_fail*', '*teardown_function(function):*', '*assert False*', '*Captured stdout*', '*teardown func*', '*test_fail*', '*def test_fail():', '*failingfunc*', '*1 failed*1 error*'])

    def test_setup_teardown_output_and_test_failure(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test for issue #442.'
        pytester.makepyfile('\n            def setup_function(function):\n                print("setup func")\n\n            def test_fail():\n                assert 0, "failingfunc"\n\n            def teardown_function(function):\n                print("teardown func")\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*test_fail*', '*def test_fail():', '*failingfunc*', '*Captured stdout setup*', '*setup func*', '*Captured stdout teardown*', '*teardown func*', '*1 failed*'])

class TestTerminalFunctional:

    def test_deselected(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        testpath = pytester.makepyfile('\n                def test_one():\n                    pass\n                def test_two():\n                    pass\n                def test_three():\n                    pass\n           ')
        result = pytester.runpytest('-k', 'test_t', testpath)
        result.stdout.fnmatch_lines(['collected 3 items / 1 deselected / 2 selected', '*test_deselected.py ..*'])
        assert result.ret == 0

    def test_deselected_with_hook_wrapper(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            import pytest\n\n            @pytest.hookimpl(wrapper=True)\n            def pytest_collection_modifyitems(config, items):\n                yield\n                deselected = items.pop()\n                config.hook.pytest_deselected(items=[deselected])\n            ')
        testpath = pytester.makepyfile('\n                def test_one():\n                    pass\n                def test_two():\n                    pass\n                def test_three():\n                    pass\n           ')
        result = pytester.runpytest(testpath)
        result.stdout.fnmatch_lines(['collected 3 items / 1 deselected / 2 selected', '*= 2 passed, 1 deselected in*'])
        assert result.ret == 0

    def test_show_deselected_items_using_markexpr_before_test_execution(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile(test_show_deselected='\n            import pytest\n\n            @pytest.mark.foo\n            def test_foobar():\n                pass\n\n            @pytest.mark.bar\n            def test_bar():\n                pass\n\n            def test_pass():\n                pass\n        ')
        result = pytester.runpytest('-m', 'not foo')
        result.stdout.fnmatch_lines(['collected 3 items / 1 deselected / 2 selected', '*test_show_deselected.py ..*', '*= 2 passed, 1 deselected in * =*'])
        result.stdout.no_fnmatch_line('*= 1 deselected =*')
        assert result.ret == 0

    def test_selected_count_with_error(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(test_selected_count_3='\n                def test_one():\n                    pass\n                def test_two():\n                    pass\n                def test_three():\n                    pass\n            ', test_selected_count_error='\n                5/0\n                def test_foo():\n                    pass\n                def test_bar():\n                    pass\n            ')
        result = pytester.runpytest('-k', 'test_t')
        result.stdout.fnmatch_lines(['collected 3 items / 1 error / 1 deselected / 2 selected', '* ERROR collecting test_selected_count_error.py *'])
        assert result.ret == ExitCode.INTERRUPTED

    def test_no_skip_summary_if_failure(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            def test_ok():\n                pass\n            def test_fail():\n                assert 0\n            def test_skip():\n                pytest.skip("dontshow")\n        ')
        result = pytester.runpytest()
        assert result.stdout.str().find('skip test summary') == -1
        assert result.ret == 1

    def test_passes(self, pytester: Pytester) -> None:
        if False:
            return 10
        p1 = pytester.makepyfile('\n            def test_passes():\n                pass\n            class TestClass(object):\n                def test_method(self):\n                    pass\n        ')
        old = p1.parent
        pytester.chdir()
        try:
            result = pytester.runpytest()
        finally:
            os.chdir(old)
        result.stdout.fnmatch_lines(['test_passes.py ..*', '* 2 pass*'])
        assert result.ret == 0

    def test_header_trailer_info(self, monkeypatch: MonkeyPatch, pytester: Pytester, request) -> None:
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD')
        pytester.makepyfile('\n            def test_passes():\n                pass\n        ')
        result = pytester.runpytest()
        verinfo = '.'.join(map(str, sys.version_info[:3]))
        result.stdout.fnmatch_lines(['*===== test session starts ====*', 'platform %s -- Python %s*pytest-%s**pluggy-%s' % (sys.platform, verinfo, pytest.__version__, pluggy.__version__), '*test_header_trailer_info.py .*', '=* 1 passed*in *.[0-9][0-9]s *='])
        if request.config.pluginmanager.list_plugin_distinfo():
            result.stdout.fnmatch_lines(['plugins: *'])

    def test_no_header_trailer_info(self, monkeypatch: MonkeyPatch, pytester: Pytester, request) -> None:
        if False:
            return 10
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD')
        pytester.makepyfile('\n            def test_passes():\n                pass\n        ')
        result = pytester.runpytest('--no-header')
        verinfo = '.'.join(map(str, sys.version_info[:3]))
        result.stdout.no_fnmatch_line('platform %s -- Python %s*pytest-%s**pluggy-%s' % (sys.platform, verinfo, pytest.__version__, pluggy.__version__))
        if request.config.pluginmanager.list_plugin_distinfo():
            result.stdout.no_fnmatch_line('plugins: *')

    def test_header(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.path.joinpath('tests').mkdir()
        pytester.path.joinpath('gui').mkdir()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['rootdir: *test_header0'])
        pytester.makeini('[pytest]')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['rootdir: *test_header0', 'configfile: tox.ini'])
        pytester.makeini('\n            [pytest]\n            testpaths = tests gui\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['rootdir: *test_header0', 'configfile: tox.ini', 'testpaths: tests, gui'])
        result = pytester.runpytest('tests')
        result.stdout.fnmatch_lines(['rootdir: *test_header0', 'configfile: tox.ini'])

    def test_header_absolute_testpath(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Regresstion test for #7814.'
        tests = pytester.path.joinpath('tests')
        tests.mkdir()
        pytester.makepyprojecttoml("\n            [tool.pytest.ini_options]\n            testpaths = ['{}']\n        ".format(tests))
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['rootdir: *absolute_testpath0', 'configfile: pyproject.toml', f'testpaths: {tests}'])

    def test_no_header(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.path.joinpath('tests').mkdir()
        pytester.path.joinpath('gui').mkdir()
        pytester.makeini('\n            [pytest]\n            testpaths = tests gui\n        ')
        result = pytester.runpytest('--no-header')
        result.stdout.no_fnmatch_line('rootdir: *test_header0, inifile: tox.ini, testpaths: tests, gui')
        result = pytester.runpytest('tests', '--no-header')
        result.stdout.no_fnmatch_line('rootdir: *test_header0, inifile: tox.ini')

    def test_no_summary(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('\n            def test_no_summary():\n                assert false\n        ')
        result = pytester.runpytest(p1, '--no-summary')
        result.stdout.no_fnmatch_line('*= FAILURES =*')

    def test_showlocals(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('\n            def test_showlocals():\n                x = 3\n                y = "x" * 5000\n                assert 0\n        ')
        result = pytester.runpytest(p1, '-l')
        result.stdout.fnmatch_lines(['x* = 3', "y* = 'xxxxxx*"])

    def test_noshowlocals_addopts_override(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makeini('[pytest]\naddopts=--showlocals')
        p1 = pytester.makepyfile('\n            def test_noshowlocals():\n                x = 3\n                y = "x" * 5000\n                assert 0\n        ')
        result = pytester.runpytest(p1, '--no-showlocals')
        result.stdout.no_fnmatch_line('x* = 3')
        result.stdout.no_fnmatch_line("y* = 'xxxxxx*")

    def test_showlocals_short(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = pytester.makepyfile('\n            def test_showlocals_short():\n                x = 3\n                y = "xxxx"\n                assert 0\n        ')
        result = pytester.runpytest(p1, '-l', '--tb=short')
        result.stdout.fnmatch_lines(['test_showlocals_short.py:*', '    assert 0', 'E   assert 0', '        x          = 3', "        y          = 'xxxx'"])

    @pytest.fixture
    def verbose_testfile(self, pytester: Pytester) -> Path:
        if False:
            return 10
        return pytester.makepyfile('\n            import pytest\n            def test_fail():\n                raise ValueError()\n            def test_pass():\n                pass\n            class TestClass(object):\n                def test_skip(self):\n                    pytest.skip("hello")\n            def test_gen():\n                def check(x):\n                    assert x == 1\n                yield check, 0\n        ')

    def test_verbose_reporting(self, verbose_testfile, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        result = pytester.runpytest(verbose_testfile, '-v', '-Walways::pytest.PytestWarning')
        result.stdout.fnmatch_lines(['*test_verbose_reporting.py::test_fail *FAIL*', '*test_verbose_reporting.py::test_pass *PASS*', '*test_verbose_reporting.py::TestClass::test_skip *SKIP*', '*test_verbose_reporting.py::test_gen *XFAIL*'])
        assert result.ret == 1

    def test_verbose_reporting_xdist(self, verbose_testfile, monkeypatch: MonkeyPatch, pytester: Pytester, pytestconfig) -> None:
        if False:
            while True:
                i = 10
        if not pytestconfig.pluginmanager.get_plugin('xdist'):
            pytest.skip('xdist plugin not installed')
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD')
        result = pytester.runpytest(verbose_testfile, '-v', '-n 1', '-Walways::pytest.PytestWarning')
        result.stdout.fnmatch_lines(['*FAIL*test_verbose_reporting_xdist.py::test_fail*'])
        assert result.ret == 1

    def test_quiet_reporting(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = pytester.makepyfile('def test_pass(): pass')
        result = pytester.runpytest(p1, '-q')
        s = result.stdout.str()
        assert 'test session starts' not in s
        assert p1.name not in s
        assert '===' not in s
        assert 'passed' in s

    def test_more_quiet_reporting(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('def test_pass(): pass')
        result = pytester.runpytest(p1, '-qq')
        s = result.stdout.str()
        assert 'test session starts' not in s
        assert p1.name not in s
        assert '===' not in s
        assert 'passed' not in s

    @pytest.mark.parametrize('params', [(), ('--collect-only',)], ids=['no-params', 'collect-only'])
    def test_report_collectionfinish_hook(self, pytester: Pytester, params) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest("\n            def pytest_report_collectionfinish(config, start_path, items):\n                return [f'hello from hook: {len(items)} items']\n        ")
        pytester.makepyfile("\n            import pytest\n            @pytest.mark.parametrize('i', range(3))\n            def test(i):\n                pass\n        ")
        result = pytester.runpytest(*params)
        result.stdout.fnmatch_lines(['collected 3 items', 'hello from hook: 3 items'])

    def test_summary_f_alias(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        "Test that 'f' and 'F' report chars are aliases and don't show up twice in the summary (#6334)"
        pytester.makepyfile('\n            def test():\n                assert False\n            ')
        result = pytester.runpytest('-rfF')
        expected = 'FAILED test_summary_f_alias.py::test - assert False'
        result.stdout.fnmatch_lines([expected])
        assert result.stdout.lines.count(expected) == 1

    def test_summary_s_alias(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        "Test that 's' and 'S' report chars are aliases and don't show up twice in the summary"
        pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.skip\n            def test():\n                pass\n            ')
        result = pytester.runpytest('-rsS')
        expected = 'SKIPPED [1] test_summary_s_alias.py:3: unconditional skip'
        result.stdout.fnmatch_lines([expected])
        assert result.stdout.lines.count(expected) == 1

@pytest.mark.parametrize(('use_ci', 'expected_message'), ((True, f"- AssertionError: {'this_failed' * 100}"), (False, '- AssertionError: this_failedt...')), ids=('on CI', 'not on CI'))
def test_fail_extra_reporting(pytester: Pytester, monkeypatch, use_ci: bool, expected_message: str) -> None:
    if False:
        while True:
            i = 10
    if use_ci:
        monkeypatch.setenv('CI', 'true')
    else:
        monkeypatch.delenv('CI', raising=False)
    monkeypatch.setenv('COLUMNS', '80')
    pytester.makepyfile("def test_this(): assert 0, 'this_failed' * 100")
    result = pytester.runpytest('-rN')
    result.stdout.no_fnmatch_line('*short test summary*')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*test summary*', f'FAILED test_fail_extra_reporting.py::test_this {expected_message}'])

def test_fail_reporting_on_pass(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('def test_this(): assert 1')
    result = pytester.runpytest('-rf')
    result.stdout.no_fnmatch_line('*short test summary*')

def test_pass_extra_reporting(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('def test_this(): assert 1')
    result = pytester.runpytest()
    result.stdout.no_fnmatch_line('*short test summary*')
    result = pytester.runpytest('-rp')
    result.stdout.fnmatch_lines(['*test summary*', 'PASS*test_pass_extra_reporting*'])

def test_pass_reporting_on_fail(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('def test_this(): assert 0')
    result = pytester.runpytest('-rp')
    result.stdout.no_fnmatch_line('*short test summary*')

def test_pass_output_reporting(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def setup_module():\n            print("setup_module")\n\n        def teardown_module():\n            print("teardown_module")\n\n        def test_pass_has_output():\n            print("Four score and seven years ago...")\n\n        def test_pass_no_output():\n            pass\n    ')
    result = pytester.runpytest()
    s = result.stdout.str()
    assert 'test_pass_has_output' not in s
    assert 'Four score and seven years ago...' not in s
    assert 'test_pass_no_output' not in s
    result = pytester.runpytest('-rPp')
    result.stdout.fnmatch_lines(['*= PASSES =*', '*_ test_pass_has_output _*', '*- Captured stdout setup -*', 'setup_module', '*- Captured stdout call -*', 'Four score and seven years ago...', '*- Captured stdout teardown -*', 'teardown_module', '*= short test summary info =*', 'PASSED test_pass_output_reporting.py::test_pass_has_output', 'PASSED test_pass_output_reporting.py::test_pass_no_output', '*= 2 passed in *'])

def test_color_yes(pytester: Pytester, color_mapping) -> None:
    if False:
        i = 10
        return i + 15
    p1 = pytester.makepyfile('\n        def fail():\n            assert 0\n\n        def test_this():\n            fail()\n        ')
    result = pytester.runpytest('--color=yes', str(p1))
    result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['{bold}=*= test session starts =*={reset}', 'collected 1 item', '', 'test_color_yes.py {red}F{reset}{red} * [100%]{reset}', '', '=*= FAILURES =*=', '{red}{bold}_*_ test_this _*_{reset}', '', '    {kw}def{hl-reset} {function}test_this{hl-reset}():{endline}', '>       fail(){endline}', '', '{bold}{red}test_color_yes.py{reset}:5: ', '_ _ * _ _*', '', '    {kw}def{hl-reset} {function}fail{hl-reset}():{endline}', '>       {kw}assert{hl-reset} {number}0{hl-reset}{endline}', '{bold}{red}E       assert 0{reset}', '', '{bold}{red}test_color_yes.py{reset}:2: AssertionError', '{red}=*= {red}{bold}1 failed{reset}{red} in *s{reset}{red} =*={reset}']))
    result = pytester.runpytest('--color=yes', '--tb=short', str(p1))
    result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['{bold}=*= test session starts =*={reset}', 'collected 1 item', '', 'test_color_yes.py {red}F{reset}{red} * [100%]{reset}', '', '=*= FAILURES =*=', '{red}{bold}_*_ test_this _*_{reset}', '{bold}{red}test_color_yes.py{reset}:5: in test_this', '    fail(){endline}', '{bold}{red}test_color_yes.py{reset}:2: in fail', '    {kw}assert{hl-reset} {number}0{hl-reset}{endline}', '{bold}{red}E   assert 0{reset}', '{red}=*= {red}{bold}1 failed{reset}{red} in *s{reset}{red} =*={reset}']))

def test_color_no(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('def test_this(): assert 1')
    result = pytester.runpytest('--color=no')
    assert 'test session starts' in result.stdout.str()
    result.stdout.no_fnmatch_line('*\x1b[1m*')

@pytest.mark.parametrize('verbose', [True, False])
def test_color_yes_collection_on_non_atty(pytester: Pytester, verbose) -> None:
    if False:
        for i in range(10):
            print('nop')
    '#1397: Skip collect progress report when working on non-terminals.'
    pytester.makepyfile("\n        import pytest\n        @pytest.mark.parametrize('i', range(10))\n        def test_this(i):\n            assert 1\n    ")
    args = ['--color=yes']
    if verbose:
        args.append('-vv')
    result = pytester.runpytest(*args)
    assert 'test session starts' in result.stdout.str()
    assert '\x1b[1m' in result.stdout.str()
    result.stdout.no_fnmatch_line('*collecting 10 items*')
    if verbose:
        assert 'collecting ...' in result.stdout.str()
    assert 'collected 10 items' in result.stdout.str()

def test_getreportopt() -> None:
    if False:
        i = 10
        return i + 15
    from _pytest.terminal import _REPORTCHARS_DEFAULT

    class FakeConfig:

        class Option:
            reportchars = _REPORTCHARS_DEFAULT
            disable_warnings = False
        option = Option()
    config = cast(Config, FakeConfig())
    assert _REPORTCHARS_DEFAULT == 'fE'
    assert getreportopt(config) == 'wfE'
    config.option.reportchars = 'sf'
    assert getreportopt(config) == 'wsf'
    config.option.reportchars = 'sfxw'
    assert getreportopt(config) == 'sfxw'
    config.option.reportchars = 'a'
    assert getreportopt(config) == 'wsxXEf'
    config.option.reportchars = 'N'
    assert getreportopt(config) == 'w'
    config.option.reportchars = 'NwfE'
    assert getreportopt(config) == 'wfE'
    config.option.reportchars = 'NfENx'
    assert getreportopt(config) == 'wx'
    config.option.disable_warnings = True
    config.option.reportchars = 'a'
    assert getreportopt(config) == 'sxXEf'
    config.option.reportchars = 'sfx'
    assert getreportopt(config) == 'sfx'
    config.option.reportchars = 'sfxw'
    assert getreportopt(config) == 'sfx'
    config.option.reportchars = 'a'
    assert getreportopt(config) == 'sxXEf'
    config.option.reportchars = 'A'
    assert getreportopt(config) == 'PpsxXEf'
    config.option.reportchars = 'AN'
    assert getreportopt(config) == ''
    config.option.reportchars = 'NwfE'
    assert getreportopt(config) == 'fE'

def test_terminalreporter_reportopt_addopts(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makeini('[pytest]\naddopts=-rs')
    pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture\n        def tr(request):\n            tr = request.config.pluginmanager.getplugin("terminalreporter")\n            return tr\n        def test_opt(tr):\n            assert tr.hasopt(\'skipped\')\n            assert not tr.hasopt(\'qwe\')\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_tbstyle_short(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    p = pytester.makepyfile('\n        import pytest\n\n        @pytest.fixture\n        def arg(request):\n            return 42\n        def test_opt(arg):\n            x = 0\n            assert x\n    ')
    result = pytester.runpytest('--tb=short')
    s = result.stdout.str()
    assert 'arg = 42' not in s
    assert 'x = 0' not in s
    result.stdout.fnmatch_lines(['*%s:8*' % p.name, '    assert x', 'E   assert*'])
    result = pytester.runpytest()
    s = result.stdout.str()
    assert 'x = 0' in s
    assert 'assert x' in s

def test_traceconfig(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    result = pytester.runpytest('--traceconfig')
    result.stdout.fnmatch_lines(['*active plugins*'])
    assert result.ret == ExitCode.NO_TESTS_COLLECTED

class TestGenericReporting:
    """Test class which can be subclassed with a different option provider to
    run e.g. distributed tests."""

    def test_collect_fail(self, pytester: Pytester, option) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('import xyz\n')
        result = pytester.runpytest(*option.args)
        result.stdout.fnmatch_lines(['ImportError while importing*', '*No module named *xyz*', '*1 error*'])

    def test_maxfailures(self, pytester: Pytester, option) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_1():\n                assert 0\n            def test_2():\n                assert 0\n            def test_3():\n                assert 0\n        ')
        result = pytester.runpytest('--maxfail=2', *option.args)
        result.stdout.fnmatch_lines(['*def test_1():*', '*def test_2():*', '*! stopping after 2 failures !*', '*2 failed*'])

    def test_maxfailures_with_interrupted(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test(request):\n                request.session.shouldstop = "session_interrupted"\n                assert 0\n        ')
        result = pytester.runpytest('--maxfail=1', '-ra')
        result.stdout.fnmatch_lines(['*= short test summary info =*', 'FAILED *', '*! stopping after 1 failures !*', '*! session_interrupted !*', '*= 1 failed in*'])

    def test_tb_option(self, pytester: Pytester, option) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n            def g():\n                raise IndexError\n            def test_func():\n                print(6*7)\n                g()  # --calling--\n        ')
        for tbopt in ['long', 'short', 'no']:
            print('testing --tb=%s...' % tbopt)
            result = pytester.runpytest('-rN', '--tb=%s' % tbopt)
            s = result.stdout.str()
            if tbopt == 'long':
                assert 'print(6*7)' in s
            else:
                assert 'print(6*7)' not in s
            if tbopt != 'no':
                assert '--calling--' in s
                assert 'IndexError' in s
            else:
                assert 'FAILURES' not in s
                assert '--calling--' not in s
                assert 'IndexError' not in s

    def test_tb_crashline(self, pytester: Pytester, option) -> None:
        if False:
            print('Hello World!')
        p = pytester.makepyfile('\n            import pytest\n            def g():\n                raise IndexError\n            def test_func1():\n                print(6*7)\n                g()  # --calling--\n            def test_func2():\n                assert 0, "hello"\n        ')
        result = pytester.runpytest('--tb=line')
        bn = p.name
        result.stdout.fnmatch_lines(['*%s:3: IndexError*' % bn, '*%s:8: AssertionError: hello*' % bn])
        s = result.stdout.str()
        assert 'def test_func2' not in s

    def test_tb_crashline_pytrace_false(self, pytester: Pytester, option) -> None:
        if False:
            return 10
        p = pytester.makepyfile("\n            import pytest\n            def test_func1():\n                pytest.fail('test_func1', pytrace=False)\n        ")
        result = pytester.runpytest('--tb=line')
        result.stdout.str()
        bn = p.name
        result.stdout.fnmatch_lines(['*%s:3: Failed: test_func1' % bn])

    def test_pytest_report_header(self, pytester: Pytester, option) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            def pytest_sessionstart(session):\n                session.config._somevalue = 42\n            def pytest_report_header(config):\n                return "hello: %s" % config._somevalue\n        ')
        pytester.mkdir('a').joinpath('conftest.py').write_text('\ndef pytest_report_header(config, start_path):\n    return ["line1", str(start_path)]\n', encoding='utf-8')
        result = pytester.runpytest('a')
        result.stdout.fnmatch_lines(['*hello: 42*', 'line1', str(pytester.path)])

    def test_show_capture(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import sys\n            import logging\n            def test_one():\n                sys.stdout.write('!This is stdout!')\n                sys.stderr.write('!This is stderr!')\n                logging.warning('!This is a warning log msg!')\n                assert False, 'Something failed'\n        ")
        result = pytester.runpytest('--tb=short')
        result.stdout.fnmatch_lines(['!This is stdout!', '!This is stderr!', '*WARNING*!This is a warning log msg!'])
        result = pytester.runpytest('--show-capture=all', '--tb=short')
        result.stdout.fnmatch_lines(['!This is stdout!', '!This is stderr!', '*WARNING*!This is a warning log msg!'])
        stdout = pytester.runpytest('--show-capture=stdout', '--tb=short').stdout.str()
        assert '!This is stderr!' not in stdout
        assert '!This is stdout!' in stdout
        assert '!This is a warning log msg!' not in stdout
        stdout = pytester.runpytest('--show-capture=stderr', '--tb=short').stdout.str()
        assert '!This is stdout!' not in stdout
        assert '!This is stderr!' in stdout
        assert '!This is a warning log msg!' not in stdout
        stdout = pytester.runpytest('--show-capture=log', '--tb=short').stdout.str()
        assert '!This is stdout!' not in stdout
        assert '!This is stderr!' not in stdout
        assert '!This is a warning log msg!' in stdout
        stdout = pytester.runpytest('--show-capture=no', '--tb=short').stdout.str()
        assert '!This is stdout!' not in stdout
        assert '!This is stderr!' not in stdout
        assert '!This is a warning log msg!' not in stdout

    def test_show_capture_with_teardown_logs(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Ensure that the capturing of teardown logs honor --show-capture setting'
        pytester.makepyfile('\n            import logging\n            import sys\n            import pytest\n\n            @pytest.fixture(scope="function", autouse="True")\n            def hook_each_test(request):\n                yield\n                sys.stdout.write("!stdout!")\n                sys.stderr.write("!stderr!")\n                logging.warning("!log!")\n\n            def test_func():\n                assert False\n        ')
        result = pytester.runpytest('--show-capture=stdout', '--tb=short').stdout.str()
        assert '!stdout!' in result
        assert '!stderr!' not in result
        assert '!log!' not in result
        result = pytester.runpytest('--show-capture=stderr', '--tb=short').stdout.str()
        assert '!stdout!' not in result
        assert '!stderr!' in result
        assert '!log!' not in result
        result = pytester.runpytest('--show-capture=log', '--tb=short').stdout.str()
        assert '!stdout!' not in result
        assert '!stderr!' not in result
        assert '!log!' in result
        result = pytester.runpytest('--show-capture=no', '--tb=short').stdout.str()
        assert '!stdout!' not in result
        assert '!stderr!' not in result
        assert '!log!' not in result

@pytest.mark.xfail("not hasattr(os, 'dup')")
def test_fdopen_kept_alive_issue124(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile("\n        import os, sys\n        k = []\n        def test_open_file_and_keep_alive(capfd):\n            stdout = os.fdopen(1, 'w', buffering=1, encoding='utf-8')\n            k.append(stdout)\n\n        def test_close_kept_alive_file():\n            stdout = k.pop()\n            stdout.close()\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*2 passed*'])

def test_tbstyle_native_setup_error(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def setup_error_fixture():\n            raise Exception("error in exception")\n\n        def test_error_fixture(setup_error_fixture):\n            pass\n    ')
    result = pytester.runpytest('--tb=native')
    result.stdout.fnmatch_lines(['*File *test_tbstyle_native_setup_error.py", line *, in setup_error_fixture*'])

def test_terminal_summary(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makeconftest('\n        def pytest_terminal_summary(terminalreporter, exitstatus):\n            w = terminalreporter\n            w.section("hello")\n            w.line("world")\n            w.line("exitstatus: {0}".format(exitstatus))\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines('\n        *==== hello ====*\n        world\n        exitstatus: 5\n    ')

@pytest.mark.filterwarnings('default::UserWarning')
def test_terminal_summary_warnings_are_displayed(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Test that warnings emitted during pytest_terminal_summary are displayed.\n    (#1305).\n    '
    pytester.makeconftest("\n        import warnings\n        def pytest_terminal_summary(terminalreporter):\n            warnings.warn(UserWarning('internal warning'))\n    ")
    pytester.makepyfile('\n        def test_failure():\n            import warnings\n            warnings.warn("warning_from_" + "test")\n            assert 0\n    ')
    result = pytester.runpytest('-ra')
    result.stdout.fnmatch_lines(['*= warnings summary =*', '*warning_from_test*', '*= short test summary info =*', '*= warnings summary (final) =*', '*conftest.py:3:*internal warning', '*== 1 failed, 2 warnings in *'])
    result.stdout.no_fnmatch_line('*None*')
    stdout = result.stdout.str()
    assert stdout.count('warning_from_test') == 1
    assert stdout.count('=== warnings summary ') == 2

@pytest.mark.filterwarnings('default::UserWarning')
def test_terminal_summary_warnings_header_once(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        def test_failure():\n            import warnings\n            warnings.warn("warning_from_" + "test")\n            assert 0\n    ')
    result = pytester.runpytest('-ra')
    result.stdout.fnmatch_lines(['*= warnings summary =*', '*warning_from_test*', '*= short test summary info =*', '*== 1 failed, 1 warning in *'])
    result.stdout.no_fnmatch_line('*None*')
    stdout = result.stdout.str()
    assert stdout.count('warning_from_test') == 1
    assert stdout.count('=== warnings summary ') == 1

@pytest.mark.filterwarnings('default')
def test_terminal_no_summary_warnings_header_once(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def test_failure():\n            import warnings\n            warnings.warn("warning_from_" + "test")\n            assert 0\n    ')
    result = pytester.runpytest('--no-summary')
    result.stdout.no_fnmatch_line('*= warnings summary =*')
    result.stdout.no_fnmatch_line('*= short test summary info =*')

@pytest.fixture(scope='session')
def tr() -> TerminalReporter:
    if False:
        while True:
            i = 10
    config = _pytest.config._prepareconfig()
    return TerminalReporter(config)

@pytest.mark.parametrize('exp_color, exp_line, stats_arg', [('red', [('1 failed', {'bold': True, 'red': True})], {'failed': [1]}), ('red', [('1 failed', {'bold': True, 'red': True}), ('1 passed', {'bold': False, 'green': True})], {'failed': [1], 'passed': [1]}), ('red', [('1 error', {'bold': True, 'red': True})], {'error': [1]}), ('red', [('2 errors', {'bold': True, 'red': True})], {'error': [1, 2]}), ('red', [('1 passed', {'bold': False, 'green': True}), ('1 error', {'bold': True, 'red': True})], {'error': [1], 'passed': [1]}), ('yellow', [('1 weird', {'bold': True, 'yellow': True})], {'weird': [1]}), ('yellow', [('1 passed', {'bold': False, 'green': True}), ('1 weird', {'bold': True, 'yellow': True})], {'weird': [1], 'passed': [1]}), ('yellow', [('1 warning', {'bold': True, 'yellow': True})], {'warnings': [1]}), ('yellow', [('1 passed', {'bold': False, 'green': True}), ('1 warning', {'bold': True, 'yellow': True})], {'warnings': [1], 'passed': [1]}), ('green', [('5 passed', {'bold': True, 'green': True})], {'passed': [1, 2, 3, 4, 5]}), ('yellow', [('1 skipped', {'bold': True, 'yellow': True})], {'skipped': [1]}), ('green', [('1 passed', {'bold': True, 'green': True}), ('1 skipped', {'bold': False, 'yellow': True})], {'skipped': [1], 'passed': [1]}), ('yellow', [('1 deselected', {'bold': True, 'yellow': True})], {'deselected': [1]}), ('green', [('1 passed', {'bold': True, 'green': True}), ('1 deselected', {'bold': False, 'yellow': True})], {'deselected': [1], 'passed': [1]}), ('yellow', [('1 xfailed', {'bold': True, 'yellow': True})], {'xfailed': [1]}), ('green', [('1 passed', {'bold': True, 'green': True}), ('1 xfailed', {'bold': False, 'yellow': True})], {'xfailed': [1], 'passed': [1]}), ('yellow', [('1 xpassed', {'bold': True, 'yellow': True})], {'xpassed': [1]}), ('yellow', [('1 passed', {'bold': False, 'green': True}), ('1 xpassed', {'bold': True, 'yellow': True})], {'xpassed': [1], 'passed': [1]}), ('yellow', [('no tests ran', {'yellow': True})], {}), ('yellow', [('no tests ran', {'yellow': True})], {'': [1]}), ('green', [('1 passed', {'bold': True, 'green': True})], {'': [1], 'passed': [1]}), ('red', [('1 failed', {'bold': True, 'red': True}), ('2 passed', {'bold': False, 'green': True}), ('3 xfailed', {'bold': False, 'yellow': True})], {'passed': [1, 2], 'failed': [1], 'xfailed': [1, 2, 3]}), ('green', [('1 passed', {'bold': True, 'green': True}), ('2 skipped', {'bold': False, 'yellow': True}), ('3 deselected', {'bold': False, 'yellow': True}), ('2 xfailed', {'bold': False, 'yellow': True})], {'passed': [1], 'skipped': [1, 2], 'deselected': [1, 2, 3], 'xfailed': [1, 2]})])
def test_summary_stats(tr: TerminalReporter, exp_line: List[Tuple[str, Dict[str, bool]]], exp_color: str, stats_arg: Dict[str, List[object]]) -> None:
    if False:
        return 10
    tr.stats = stats_arg

    class fake_session:
        testscollected = 0
    tr._session = fake_session
    assert tr._is_last_item
    tr._main_color = None
    print('Based on stats: %s' % stats_arg)
    print(f'Expect summary: "{exp_line}"; with color "{exp_color}"')
    (line, color) = tr.build_summary_stats_line()
    print(f'Actually got:   "{line}"; with color "{color}"')
    assert line == exp_line
    assert color == exp_color

def test_skip_counting_towards_summary(tr):
    if False:
        return 10

    class DummyReport(BaseReport):
        count_towards_summary = True
    r1 = DummyReport()
    r2 = DummyReport()
    tr.stats = {'failed': (r1, r2)}
    tr._main_color = None
    res = tr.build_summary_stats_line()
    assert res == ([('2 failed', {'bold': True, 'red': True})], 'red')
    r1.count_towards_summary = False
    tr.stats = {'failed': (r1, r2)}
    tr._main_color = None
    res = tr.build_summary_stats_line()
    assert res == ([('1 failed', {'bold': True, 'red': True})], 'red')

class TestClassicOutputStyle:
    """Ensure classic output style works as expected (#3883)"""

    @pytest.fixture
    def test_files(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile(**{'test_one.py': 'def test_one(): pass', 'test_two.py': 'def test_two(): assert 0', 'sub/test_three.py': '\n                    def test_three_1(): pass\n                    def test_three_2(): assert 0\n                    def test_three_3(): pass\n                '})

    def test_normal_verbosity(self, pytester: Pytester, test_files) -> None:
        if False:
            return 10
        result = pytester.runpytest('-o', 'console_output_style=classic')
        result.stdout.fnmatch_lines(['test_one.py .', 'test_two.py F', f'sub{os.sep}test_three.py .F.', '*2 failed, 3 passed in*'])

    def test_verbose(self, pytester: Pytester, test_files) -> None:
        if False:
            return 10
        result = pytester.runpytest('-o', 'console_output_style=classic', '-v')
        result.stdout.fnmatch_lines(['test_one.py::test_one PASSED', 'test_two.py::test_two FAILED', f'sub{os.sep}test_three.py::test_three_1 PASSED', f'sub{os.sep}test_three.py::test_three_2 FAILED', f'sub{os.sep}test_three.py::test_three_3 PASSED', '*2 failed, 3 passed in*'])

    def test_quiet(self, pytester: Pytester, test_files) -> None:
        if False:
            return 10
        result = pytester.runpytest('-o', 'console_output_style=classic', '-q')
        result.stdout.fnmatch_lines(['.F.F.', '*2 failed, 3 passed in*'])

class TestProgressOutputStyle:

    @pytest.fixture
    def many_tests_files(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(test_bar="\n                import pytest\n                @pytest.mark.parametrize('i', range(10))\n                def test_bar(i): pass\n            ", test_foo="\n                import pytest\n                @pytest.mark.parametrize('i', range(5))\n                def test_foo(i): pass\n            ", test_foobar="\n                import pytest\n                @pytest.mark.parametrize('i', range(5))\n                def test_foobar(i): pass\n            ")

    def test_zero_tests_collected(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Some plugins (testmon for example) might issue pytest_runtest_logreport without any tests being\n        actually collected (#2971).'
        pytester.makeconftest("\n        def pytest_collection_modifyitems(items, config):\n            from _pytest.runner import CollectReport\n            for node_id in ('nodeid1', 'nodeid2'):\n                rep = CollectReport(node_id, 'passed', None, None)\n                rep.when = 'passed'\n                rep.duration = 0.1\n                config.hook.pytest_runtest_logreport(report=rep)\n        ")
        output = pytester.runpytest()
        output.stdout.no_fnmatch_line('*ZeroDivisionError*')
        output.stdout.fnmatch_lines(['=* 2 passed in *='])

    def test_normal(self, many_tests_files, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        output = pytester.runpytest()
        output.stdout.re_match_lines(['test_bar.py \\.{10} \\s+ \\[ 50%\\]', 'test_foo.py \\.{5} \\s+ \\[ 75%\\]', 'test_foobar.py \\.{5} \\s+ \\[100%\\]'])

    def test_colored_progress(self, pytester: Pytester, monkeypatch, color_mapping) -> None:
        if False:
            print('Hello World!')
        monkeypatch.setenv('PY_COLORS', '1')
        pytester.makepyfile(test_axfail='\n                import pytest\n                @pytest.mark.xfail\n                def test_axfail(): assert 0\n            ', test_bar="\n                import pytest\n                @pytest.mark.parametrize('i', range(10))\n                def test_bar(i): pass\n            ", test_foo='\n                import pytest\n                import warnings\n                @pytest.mark.parametrize(\'i\', range(5))\n                def test_foo(i):\n                    warnings.warn(DeprecationWarning("collection"))\n                    pass\n            ', test_foobar="\n                import pytest\n                @pytest.mark.parametrize('i', range(5))\n                def test_foobar(i): raise ValueError()\n            ")
        result = pytester.runpytest()
        result.stdout.re_match_lines(color_mapping.format_for_rematch(['test_axfail.py {yellow}x{reset}{green} \\s+ \\[  4%\\]{reset}', 'test_bar.py ({green}\\.{reset}){{10}}{green} \\s+ \\[ 52%\\]{reset}', 'test_foo.py ({green}\\.{reset}){{5}}{yellow} \\s+ \\[ 76%\\]{reset}', 'test_foobar.py ({red}F{reset}){{5}}{red} \\s+ \\[100%\\]{reset}']))
        result = pytester.runpytest('test_axfail.py')
        result.stdout.re_match_lines(color_mapping.format_for_rematch(['test_axfail.py {yellow}x{reset}{yellow} \\s+ \\[100%\\]{reset}', '^{yellow}=+ ({yellow}{bold}|{bold}{yellow})1 xfailed{reset}{yellow} in ']))

    def test_count(self, many_tests_files, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeini('\n            [pytest]\n            console_output_style = count\n        ')
        output = pytester.runpytest()
        output.stdout.re_match_lines(['test_bar.py \\.{10} \\s+ \\[10/20\\]', 'test_foo.py \\.{5} \\s+ \\[15/20\\]', 'test_foobar.py \\.{5} \\s+ \\[20/20\\]'])

    def test_verbose(self, many_tests_files, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        output = pytester.runpytest('-v')
        output.stdout.re_match_lines(['test_bar.py::test_bar\\[0\\] PASSED \\s+ \\[  5%\\]', 'test_foo.py::test_foo\\[4\\] PASSED \\s+ \\[ 75%\\]', 'test_foobar.py::test_foobar\\[4\\] PASSED \\s+ \\[100%\\]'])

    def test_verbose_count(self, many_tests_files, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeini('\n            [pytest]\n            console_output_style = count\n        ')
        output = pytester.runpytest('-v')
        output.stdout.re_match_lines(['test_bar.py::test_bar\\[0\\] PASSED \\s+ \\[ 1/20\\]', 'test_foo.py::test_foo\\[4\\] PASSED \\s+ \\[15/20\\]', 'test_foobar.py::test_foobar\\[4\\] PASSED \\s+ \\[20/20\\]'])

    def test_xdist_normal(self, many_tests_files, pytester: Pytester, monkeypatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytest.importorskip('xdist')
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', raising=False)
        output = pytester.runpytest('-n2')
        output.stdout.re_match_lines(['\\.{20} \\s+ \\[100%\\]'])

    def test_xdist_normal_count(self, many_tests_files, pytester: Pytester, monkeypatch) -> None:
        if False:
            return 10
        pytest.importorskip('xdist')
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', raising=False)
        pytester.makeini('\n            [pytest]\n            console_output_style = count\n        ')
        output = pytester.runpytest('-n2')
        output.stdout.re_match_lines(['\\.{20} \\s+ \\[20/20\\]'])

    def test_xdist_verbose(self, many_tests_files, pytester: Pytester, monkeypatch) -> None:
        if False:
            i = 10
            return i + 15
        pytest.importorskip('xdist')
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', raising=False)
        output = pytester.runpytest('-n2', '-v')
        output.stdout.re_match_lines_random(['\\[gw\\d\\] \\[\\s*\\d+%\\] PASSED test_bar.py::test_bar\\[1\\]', '\\[gw\\d\\] \\[\\s*\\d+%\\] PASSED test_foo.py::test_foo\\[1\\]', '\\[gw\\d\\] \\[\\s*\\d+%\\] PASSED test_foobar.py::test_foobar\\[1\\]'])
        output.stdout.fnmatch_lines_random([line.translate(TRANS_FNMATCH) for line in ['test_bar.py::test_bar[0] ', 'test_foo.py::test_foo[0] ', 'test_foobar.py::test_foobar[0] ', '[gw?] [  5%] PASSED test_*[?] ', '[gw?] [ 10%] PASSED test_*[?] ', '[gw?] [ 55%] PASSED test_*[?] ', '[gw?] [ 60%] PASSED test_*[?] ', '[gw?] [ 95%] PASSED test_*[?] ', '[gw?] [100%] PASSED test_*[?] ']])

    def test_capture_no(self, many_tests_files, pytester: Pytester) -> None:
        if False:
            return 10
        output = pytester.runpytest('-s')
        output.stdout.re_match_lines(['test_bar.py \\.{10}', 'test_foo.py \\.{5}', 'test_foobar.py \\.{5}'])
        output = pytester.runpytest('--capture=no')
        output.stdout.no_fnmatch_line('*%]*')

    def test_capture_no_progress_enabled(self, many_tests_files, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makeini('\n            [pytest]\n            console_output_style = progress-even-when-capture-no\n        ')
        output = pytester.runpytest('-s')
        output.stdout.re_match_lines(['test_bar.py \\.{10} \\s+ \\[ 50%\\]', 'test_foo.py \\.{5} \\s+ \\[ 75%\\]', 'test_foobar.py \\.{5} \\s+ \\[100%\\]'])

class TestProgressWithTeardown:
    """Ensure we show the correct percentages for tests that fail during teardown (#3088)"""

    @pytest.fixture
    def contest_with_teardown_fixture(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            import pytest\n\n            @pytest.fixture\n            def fail_teardown():\n                yield\n                assert False\n        ')

    @pytest.fixture
    def many_files(self, pytester: Pytester, contest_with_teardown_fixture) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(test_bar="\n                import pytest\n                @pytest.mark.parametrize('i', range(5))\n                def test_bar(fail_teardown, i):\n                    pass\n            ", test_foo="\n                import pytest\n                @pytest.mark.parametrize('i', range(15))\n                def test_foo(fail_teardown, i):\n                    pass\n            ")

    def test_teardown_simple(self, pytester: Pytester, contest_with_teardown_fixture) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            def test_foo(fail_teardown):\n                pass\n        ')
        output = pytester.runpytest()
        output.stdout.re_match_lines(['test_teardown_simple.py \\.E\\s+\\[100%\\]'])

    def test_teardown_with_test_also_failing(self, pytester: Pytester, contest_with_teardown_fixture) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_foo(fail_teardown):\n                assert 0\n        ')
        output = pytester.runpytest('-rfE')
        output.stdout.re_match_lines(['test_teardown_with_test_also_failing.py FE\\s+\\[100%\\]', 'FAILED test_teardown_with_test_also_failing.py::test_foo - assert 0', 'ERROR test_teardown_with_test_also_failing.py::test_foo - assert False'])

    def test_teardown_many(self, pytester: Pytester, many_files) -> None:
        if False:
            while True:
                i = 10
        output = pytester.runpytest()
        output.stdout.re_match_lines(['test_bar.py (\\.E){5}\\s+\\[ 25%\\]', 'test_foo.py (\\.E){15}\\s+\\[100%\\]'])

    def test_teardown_many_verbose(self, pytester: Pytester, many_files, color_mapping) -> None:
        if False:
            return 10
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['test_bar.py::test_bar[0] PASSED  * [  5%]', 'test_bar.py::test_bar[0] ERROR   * [  5%]', 'test_bar.py::test_bar[4] PASSED  * [ 25%]', 'test_foo.py::test_foo[14] PASSED * [100%]', 'test_foo.py::test_foo[14] ERROR  * [100%]', '=* 20 passed, 20 errors in *']))

    def test_xdist_normal(self, many_files, pytester: Pytester, monkeypatch) -> None:
        if False:
            print('Hello World!')
        pytest.importorskip('xdist')
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', raising=False)
        output = pytester.runpytest('-n2')
        output.stdout.re_match_lines(['[\\.E]{40} \\s+ \\[100%\\]'])

def test_skip_reasons_folding() -> None:
    if False:
        return 10
    path = 'xyz'
    lineno = 3
    message = 'justso'
    longrepr = (path, lineno, message)

    class X:
        pass
    ev1 = cast(CollectReport, X())
    ev1.when = 'execute'
    ev1.skipped = True
    ev1.longrepr = longrepr
    ev2 = cast(CollectReport, X())
    ev2.when = 'execute'
    ev2.longrepr = longrepr
    ev2.skipped = True
    ev3 = cast(CollectReport, X())
    ev3.when = 'collect'
    ev3.longrepr = longrepr
    ev3.skipped = True
    values = _folded_skips(Path.cwd(), [ev1, ev2, ev3])
    assert len(values) == 1
    (num, fspath, lineno_, reason) = values[0]
    assert num == 3
    assert fspath == path
    assert lineno_ == lineno
    assert reason == message

def test_line_with_reprcrash(monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    mocked_verbose_word = 'FAILED'
    mocked_pos = 'some::nodeid'

    def mock_get_pos(*args):
        if False:
            for i in range(10):
                print('nop')
        return mocked_pos
    monkeypatch.setattr(_pytest.terminal, '_get_node_id_with_markup', mock_get_pos)

    class config:
        pass

    class rep:

        def _get_verbose_word(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            return mocked_verbose_word

        class longrepr:

            class reprcrash:
                pass

    def check(msg, width, expected):
        if False:
            for i in range(10):
                print('nop')

        class DummyTerminalWriter:
            fullwidth = width

            def markup(self, word: str, **markup: str):
                if False:
                    for i in range(10):
                        print('nop')
                return word
        __tracebackhide__ = True
        if msg:
            rep.longrepr.reprcrash.message = msg
        actual = _get_line_with_reprcrash_message(config, rep(), DummyTerminalWriter(), {})
        assert actual == expected
        if actual != f'{mocked_verbose_word} {mocked_pos}':
            assert len(actual) <= width
            assert wcswidth(actual) <= width
    check(None, 80, 'FAILED some::nodeid')
    check('msg', 80, 'FAILED some::nodeid - msg')
    check('msg', 3, 'FAILED some::nodeid')
    check('msg', 24, 'FAILED some::nodeid')
    check('msg', 25, 'FAILED some::nodeid - msg')
    check('some longer msg', 24, 'FAILED some::nodeid')
    check('some longer msg', 25, 'FAILED some::nodeid - ...')
    check('some longer msg', 26, 'FAILED some::nodeid - s...')
    check('some\nmessage', 25, 'FAILED some::nodeid - ...')
    check('some\nmessage', 26, 'FAILED some::nodeid - some')
    check('some\nmessage', 80, 'FAILED some::nodeid - some')
    check('🉐🉐🉐🉐🉐\n2nd line', 25, 'FAILED some::nodeid - ...')
    check('🉐🉐🉐🉐🉐\n2nd line', 26, 'FAILED some::nodeid - ...')
    check('🉐🉐🉐🉐🉐\n2nd line', 27, 'FAILED some::nodeid - 🉐...')
    check('🉐🉐🉐🉐🉐\n2nd line', 28, 'FAILED some::nodeid - 🉐...')
    check('🉐🉐🉐🉐🉐\n2nd line', 29, 'FAILED some::nodeid - 🉐🉐...')
    mocked_pos = 'nodeid::🉐::withunicode'
    check('🉐🉐🉐🉐🉐\n2nd line', 29, 'FAILED nodeid::🉐::withunicode')
    check('🉐🉐🉐🉐🉐\n2nd line', 40, 'FAILED nodeid::🉐::withunicode - 🉐🉐...')
    check('🉐🉐🉐🉐🉐\n2nd line', 41, 'FAILED nodeid::🉐::withunicode - 🉐🉐...')
    check('🉐🉐🉐🉐🉐\n2nd line', 42, 'FAILED nodeid::🉐::withunicode - 🉐🉐🉐...')
    check('🉐🉐🉐🉐🉐\n2nd line', 80, 'FAILED nodeid::🉐::withunicode - 🉐🉐🉐🉐🉐')

@pytest.mark.parametrize('seconds, expected', [(10.0, '10.00s'), (10.34, '10.34s'), (59.99, '59.99s'), (60.55, '60.55s (0:01:00)'), (123.55, '123.55s (0:02:03)'), (60 * 60 + 0.5, '3600.50s (1:00:00)')])
def test_format_session_duration(seconds, expected):
    if False:
        for i in range(10):
            print('nop')
    from _pytest.terminal import format_session_duration
    assert format_session_duration(seconds) == expected

def test_collecterror(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    p1 = pytester.makepyfile('raise SyntaxError()')
    result = pytester.runpytest('-ra', str(p1))
    result.stdout.fnmatch_lines(['collected 0 items / 1 error', '*= ERRORS =*', '*_ ERROR collecting test_collecterror.py _*', 'E   SyntaxError: *', '*= short test summary info =*', 'ERROR test_collecterror.py', '*! Interrupted: 1 error during collection !*', '*= 1 error in *'])

def test_no_summary_collecterror(pytester: Pytester) -> None:
    if False:
        return 10
    p1 = pytester.makepyfile('raise SyntaxError()')
    result = pytester.runpytest('-ra', '--no-summary', str(p1))
    result.stdout.no_fnmatch_line('*= ERRORS =*')

def test_via_exec(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    p1 = pytester.makepyfile("exec('def test_via_exec(): pass')")
    result = pytester.runpytest(str(p1), '-vv')
    result.stdout.fnmatch_lines(['test_via_exec.py::test_via_exec <- <string> PASSED*', '*= 1 passed in *'])

class TestCodeHighlight:

    def test_code_highlight_simple(self, pytester: Pytester, color_mapping) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 10\n        ')
        result = pytester.runpytest('--color=yes')
        result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['    {kw}def{hl-reset} {function}test_foo{hl-reset}():{endline}', '>       {kw}assert{hl-reset} {number}1{hl-reset} == {number}10{hl-reset}{endline}', '{bold}{red}E       assert 1 == 10{reset}']))

    def test_code_highlight_continuation(self, pytester: Pytester, color_mapping) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile("\n            def test_foo():\n                print('''\n                '''); assert 0\n        ")
        result = pytester.runpytest('--color=yes')
        result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['    {kw}def{hl-reset} {function}test_foo{hl-reset}():{endline}', "        {print}print{hl-reset}({str}'''{hl-reset}{str}{hl-reset}", ">   {str}    {hl-reset}{str}'''{hl-reset}); {kw}assert{hl-reset} {number}0{hl-reset}{endline}", '{bold}{red}E       assert 0{reset}']))

    def test_code_highlight_custom_theme(self, pytester: Pytester, color_mapping, monkeypatch: MonkeyPatch) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 10\n        ')
        monkeypatch.setenv('PYTEST_THEME', 'solarized-dark')
        monkeypatch.setenv('PYTEST_THEME_MODE', 'dark')
        result = pytester.runpytest('--color=yes')
        result.stdout.fnmatch_lines(color_mapping.format_for_fnmatch(['    {kw}def{hl-reset} {function}test_foo{hl-reset}():{endline}', '>       {kw}assert{hl-reset} {number}1{hl-reset} == {number}10{hl-reset}{endline}', '{bold}{red}E       assert 1 == 10{reset}']))

    def test_code_highlight_invalid_theme(self, pytester: Pytester, color_mapping, monkeypatch: MonkeyPatch) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 10\n        ')
        monkeypatch.setenv('PYTEST_THEME', 'invalid')
        result = pytester.runpytest_subprocess('--color=yes')
        result.stderr.fnmatch_lines("ERROR: PYTEST_THEME environment variable had an invalid value: 'invalid'. Only valid pygment styles are allowed.")

    def test_code_highlight_invalid_theme_mode(self, pytester: Pytester, color_mapping, monkeypatch: MonkeyPatch) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 10\n        ')
        monkeypatch.setenv('PYTEST_THEME_MODE', 'invalid')
        result = pytester.runpytest_subprocess('--color=yes')
        result.stderr.fnmatch_lines("ERROR: PYTEST_THEME_MODE environment variable had an invalid value: 'invalid'. The only allowed values are 'dark' and 'light'.")

def test_raw_skip_reason_skipped() -> None:
    if False:
        while True:
            i = 10
    report = SimpleNamespace()
    report.skipped = True
    report.longrepr = ('xyz', 3, 'Skipped: Just so')
    reason = _get_raw_skip_reason(cast(TestReport, report))
    assert reason == 'Just so'

def test_raw_skip_reason_xfail() -> None:
    if False:
        for i in range(10):
            print('nop')
    report = SimpleNamespace()
    report.wasxfail = 'reason: To everything there is a season'
    reason = _get_raw_skip_reason(cast(TestReport, report))
    assert reason == 'To everything there is a season'

def test_format_trimmed() -> None:
    if False:
        print('Hello World!')
    msg = 'unconditional skip'
    assert _format_trimmed(' ({}) ', msg, len(msg) + 4) == ' (unconditional skip) '
    assert _format_trimmed(' ({}) ', msg, len(msg) + 3) == ' (unconditional ...) '