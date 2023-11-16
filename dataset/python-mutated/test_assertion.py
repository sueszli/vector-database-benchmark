import collections
import sys
import textwrap
from typing import Any
from typing import List
from typing import MutableSequence
from typing import Optional
import attr
import _pytest.assertion as plugin
import pytest
from _pytest import outcomes
from _pytest.assertion import truncate
from _pytest.assertion import util
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester

def mock_config(verbose=0):
    if False:
        for i in range(10):
            print('nop')

    class TerminalWriter:

        def _highlight(self, source, lexer):
            if False:
                for i in range(10):
                    print('nop')
            return source

    class Config:

        def getoption(self, name):
            if False:
                print('Hello World!')
            if name == 'verbose':
                return verbose
            raise KeyError('Not mocked out: %s' % name)

        def get_terminal_writer(self):
            if False:
                return 10
            return TerminalWriter()
    return Config()

class TestImportHookInstallation:

    @pytest.mark.parametrize('initial_conftest', [True, False])
    @pytest.mark.parametrize('mode', ['plain', 'rewrite'])
    def test_conftest_assertion_rewrite(self, pytester: Pytester, initial_conftest, mode) -> None:
        if False:
            i = 10
            return i + 15
        'Test that conftest files are using assertion rewrite on import (#1619).'
        pytester.mkdir('foo')
        pytester.mkdir('foo/tests')
        conftest_path = 'conftest.py' if initial_conftest else 'foo/conftest.py'
        contents = {conftest_path: '\n                import pytest\n                @pytest.fixture\n                def check_first():\n                    def check(values, value):\n                        assert values.pop(0) == value\n                    return check\n            ', 'foo/tests/test_foo.py': '\n                def test(check_first):\n                    check_first([10, 30], 30)\n            '}
        pytester.makepyfile(**contents)
        result = pytester.runpytest_subprocess('--assert=%s' % mode)
        if mode == 'plain':
            expected = 'E       AssertionError'
        elif mode == 'rewrite':
            expected = '*assert 10 == 30*'
        else:
            assert 0
        result.stdout.fnmatch_lines([expected])

    def test_rewrite_assertions_pytester_plugin(self, pytester: Pytester) -> None:
        if False:
            return 10
        '\n        Assertions in the pytester plugin must also benefit from assertion\n        rewriting (#1920).\n        '
        pytester.makepyfile("\n            pytest_plugins = ['pytester']\n            def test_dummy_failure(pytester):  # how meta!\n                pytester.makepyfile('def test(): assert 0')\n                r = pytester.inline_run()\n                r.assertoutcome(passed=1)\n        ")
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(['>       r.assertoutcome(passed=1)', 'E       AssertionError: ([[][]], [[][]], [[]<TestReport *>[]])*', "E       assert {'failed': 1,... 'skipped': 0} == {'failed': 0,... 'skipped': 0}", 'E         Omitting 1 identical items, use -vv to show', 'E         Differing items:', 'E         Use -v to get more diff'])
        result.stdout.fnmatch_lines_random(["E         {'failed': 1} != {'failed': 0}", "E         {'passed': 0} != {'passed': 1}"])

    @pytest.mark.parametrize('mode', ['plain', 'rewrite'])
    def test_pytest_plugins_rewrite(self, pytester: Pytester, mode) -> None:
        if False:
            print('Hello World!')
        contents = {'conftest.py': "\n                pytest_plugins = ['ham']\n            ", 'ham.py': '\n                import pytest\n                @pytest.fixture\n                def check_first():\n                    def check(values, value):\n                        assert values.pop(0) == value\n                    return check\n            ', 'test_foo.py': '\n                def test_foo(check_first):\n                    check_first([10, 30], 30)\n            '}
        pytester.makepyfile(**contents)
        result = pytester.runpytest_subprocess('--assert=%s' % mode)
        if mode == 'plain':
            expected = 'E       AssertionError'
        elif mode == 'rewrite':
            expected = '*assert 10 == 30*'
        else:
            assert 0
        result.stdout.fnmatch_lines([expected])

    @pytest.mark.parametrize('mode', ['str', 'list'])
    def test_pytest_plugins_rewrite_module_names(self, pytester: Pytester, mode) -> None:
        if False:
            print('Hello World!')
        'Test that pluginmanager correct marks pytest_plugins variables\n        for assertion rewriting if they are defined as plain strings or\n        list of strings (#1888).\n        '
        plugins = '"ham"' if mode == 'str' else '["ham"]'
        contents = {'conftest.py': '\n                pytest_plugins = {plugins}\n            '.format(plugins=plugins), 'ham.py': '\n                import pytest\n            ', 'test_foo.py': "\n                def test_foo(pytestconfig):\n                    assert 'ham' in pytestconfig.pluginmanager.rewrite_hook._must_rewrite\n            "}
        pytester.makepyfile(**contents)
        result = pytester.runpytest_subprocess('--assert=rewrite')
        assert result.ret == 0

    def test_pytest_plugins_rewrite_module_names_correctly(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        'Test that we match files correctly when they are marked for rewriting (#2939).'
        contents = {'conftest.py': '                pytest_plugins = "ham"\n            ', 'ham.py': '', 'hamster.py': '', 'test_foo.py': "                def test_foo(pytestconfig):\n                    assert pytestconfig.pluginmanager.rewrite_hook.find_spec('ham') is not None\n                    assert pytestconfig.pluginmanager.rewrite_hook.find_spec('hamster') is None\n            "}
        pytester.makepyfile(**contents)
        result = pytester.runpytest_subprocess('--assert=rewrite')
        assert result.ret == 0

    @pytest.mark.parametrize('mode', ['plain', 'rewrite'])
    def test_installed_plugin_rewrite(self, pytester: Pytester, mode, monkeypatch) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', raising=False)
        pytester.mkdir('hampkg')
        contents = {'hampkg/__init__.py': '                import pytest\n\n                @pytest.fixture\n                def check_first2():\n                    def check(values, value):\n                        assert values.pop(0) == value\n                    return check\n            ', 'spamplugin.py': '            import pytest\n            from hampkg import check_first2\n\n            @pytest.fixture\n            def check_first():\n                def check(values, value):\n                    assert values.pop(0) == value\n                return check\n            ', 'mainwrapper.py': "            import importlib.metadata\n            import pytest\n\n            class DummyEntryPoint(object):\n                name = 'spam'\n                module_name = 'spam.py'\n                group = 'pytest11'\n\n                def load(self):\n                    import spamplugin\n                    return spamplugin\n\n            class DummyDistInfo(object):\n                version = '1.0'\n                files = ('spamplugin.py', 'hampkg/__init__.py')\n                entry_points = (DummyEntryPoint(),)\n                metadata = {'name': 'foo'}\n\n            def distributions():\n                return (DummyDistInfo(),)\n\n            importlib.metadata.distributions = distributions\n            pytest.main()\n            ", 'test_foo.py': '            def test(check_first):\n                check_first([10, 30], 30)\n\n            def test2(check_first2):\n                check_first([10, 30], 30)\n            '}
        pytester.makepyfile(**contents)
        result = pytester.run(sys.executable, 'mainwrapper.py', '-s', '--assert=%s' % mode)
        if mode == 'plain':
            expected = 'E       AssertionError'
        elif mode == 'rewrite':
            expected = '*assert 10 == 30*'
        else:
            assert 0
        result.stdout.fnmatch_lines([expected])

    def test_rewrite_ast(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.mkdir('pkg')
        contents = {'pkg/__init__.py': "\n                import pytest\n                pytest.register_assert_rewrite('pkg.helper')\n            ", 'pkg/helper.py': '\n                def tool():\n                    a, b = 2, 3\n                    assert a == b\n            ', 'pkg/plugin.py': '\n                import pytest, pkg.helper\n                @pytest.fixture\n                def tool():\n                    return pkg.helper.tool\n            ', 'pkg/other.py': '\n                values = [3, 2]\n                def tool():\n                    assert values.pop() == 3\n            ', 'conftest.py': "\n                pytest_plugins = ['pkg.plugin']\n            ", 'test_pkg.py': '\n                import pkg.other\n                def test_tool(tool):\n                    tool()\n                def test_other():\n                    pkg.other.tool()\n            '}
        pytester.makepyfile(**contents)
        result = pytester.runpytest_subprocess('--assert=rewrite')
        result.stdout.fnmatch_lines(['>*assert a == b*', 'E*assert 2 == 3*', '>*assert values.pop() == 3*', 'E*AssertionError'])

    def test_register_assert_rewrite_checks_types(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError):
            pytest.register_assert_rewrite(['pytest_tests_internal_non_existing'])
        pytest.register_assert_rewrite('pytest_tests_internal_non_existing', 'pytest_tests_internal_non_existing2')

class TestBinReprIntegration:

    def test_pytest_assertrepr_compare_called(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeconftest('\n            import pytest\n            values = []\n            def pytest_assertrepr_compare(op, left, right):\n                values.append((op, left, right))\n\n            @pytest.fixture\n            def list(request):\n                return values\n        ')
        pytester.makepyfile('\n            def test_hello():\n                assert 0 == 1\n            def test_check(list):\n                assert list == [("==", 0, 1)]\n        ')
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_hello*FAIL*', '*test_check*PASS*'])

def callop(op: str, left: Any, right: Any, verbose: int=0) -> Optional[List[str]]:
    if False:
        return 10
    config = mock_config(verbose=verbose)
    return plugin.pytest_assertrepr_compare(config, op, left, right)

def callequal(left: Any, right: Any, verbose: int=0) -> Optional[List[str]]:
    if False:
        return 10
    return callop('==', left, right, verbose)

class TestAssert_reprcompare:

    def test_different_types(self) -> None:
        if False:
            while True:
                i = 10
        assert callequal([0, 1], 'foo') is None

    def test_summary(self) -> None:
        if False:
            return 10
        lines = callequal([0, 1], [0, 2])
        assert lines is not None
        summary = lines[0]
        assert len(summary) < 65

    def test_text_diff(self) -> None:
        if False:
            print('Hello World!')
        assert callequal('spam', 'eggs') == ["'spam' == 'eggs'", '- eggs', '+ spam']

    def test_text_skipping(self) -> None:
        if False:
            while True:
                i = 10
        lines = callequal('a' * 50 + 'spam', 'a' * 50 + 'eggs')
        assert lines is not None
        assert 'Skipping' in lines[1]
        for line in lines:
            assert 'a' * 50 not in line

    def test_text_skipping_verbose(self) -> None:
        if False:
            i = 10
            return i + 15
        lines = callequal('a' * 50 + 'spam', 'a' * 50 + 'eggs', verbose=1)
        assert lines is not None
        assert '- ' + 'a' * 50 + 'eggs' in lines
        assert '+ ' + 'a' * 50 + 'spam' in lines

    def test_multiline_text_diff(self) -> None:
        if False:
            while True:
                i = 10
        left = 'foo\nspam\nbar'
        right = 'foo\neggs\nbar'
        diff = callequal(left, right)
        assert diff is not None
        assert '- eggs' in diff
        assert '+ spam' in diff

    def test_bytes_diff_normal(self) -> None:
        if False:
            return 10
        'Check special handling for bytes diff (#5260)'
        diff = callequal(b'spam', b'eggs')
        assert diff == ["b'spam' == b'eggs'", "At index 0 diff: b's' != b'e'", 'Use -v to get more diff']

    def test_bytes_diff_verbose(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check special handling for bytes diff (#5260)'
        diff = callequal(b'spam', b'eggs', verbose=1)
        assert diff == ["b'spam' == b'eggs'", "At index 0 diff: b's' != b'e'", 'Full diff:', "- b'eggs'", "+ b'spam'"]

    def test_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = callequal([0, 1], [0, 2])
        assert expl is not None
        assert len(expl) > 1

    @pytest.mark.parametrize(['left', 'right', 'expected'], [pytest.param([0, 1], [0, 2], '\n                Full diff:\n                - [0, 2]\n                ?     ^\n                + [0, 1]\n                ?     ^\n            ', id='lists'), pytest.param({0: 1}, {0: 2}, '\n                Full diff:\n                - {0: 2}\n                ?     ^\n                + {0: 1}\n                ?     ^\n            ', id='dicts'), pytest.param({0, 1}, {0, 2}, '\n                Full diff:\n                - {0, 2}\n                ?     ^\n                + {0, 1}\n                ?     ^\n            ', id='sets')])
    def test_iterable_full_diff(self, left, right, expected) -> None:
        if False:
            i = 10
            return i + 15
        'Test the full diff assertion failure explanation.\n\n        When verbose is False, then just a -v notice to get the diff is rendered,\n        when verbose is True, then ndiff of the pprint is returned.\n        '
        expl = callequal(left, right, verbose=0)
        assert expl is not None
        assert expl[-1] == 'Use -v to get more diff'
        verbose_expl = callequal(left, right, verbose=1)
        assert verbose_expl is not None
        assert '\n'.join(verbose_expl).endswith(textwrap.dedent(expected).strip())

    def test_iterable_quiet(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = callequal([1, 2], [10, 2], verbose=-1)
        assert expl == ['[1, 2] == [10, 2]', 'At index 0 diff: 1 != 10', 'Use -v to get more diff']

    def test_iterable_full_diff_ci(self, monkeypatch: MonkeyPatch, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_full_diff():\n                left = [0, 1]\n                right = [0, 2]\n                assert left == right\n        ')
        monkeypatch.setenv('CI', 'true')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['E         Full diff:'])
        monkeypatch.delenv('CI', raising=False)
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['E         Use -v to get more diff'])

    def test_list_different_lengths(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = callequal([0, 1], [0, 1, 2])
        assert expl is not None
        assert len(expl) > 1
        expl = callequal([0, 1, 2], [0, 1])
        assert expl is not None
        assert len(expl) > 1

    def test_list_wrap_for_multiple_lines(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        long_d = 'd' * 80
        l1 = ['a', 'b', 'c']
        l2 = ['a', 'b', 'c', long_d]
        diff = callequal(l1, l2, verbose=True)
        assert diff == ["['a', 'b', 'c'] == ['a', 'b', 'c...dddddddddddd']", "Right contains one more item: '" + long_d + "'", 'Full diff:', '  [', "   'a',", "   'b',", "   'c',", "-  '" + long_d + "',", '  ]']
        diff = callequal(l2, l1, verbose=True)
        assert diff == ["['a', 'b', 'c...dddddddddddd'] == ['a', 'b', 'c']", "Left contains one more item: '" + long_d + "'", 'Full diff:', '  [', "   'a',", "   'b',", "   'c',", "+  '" + long_d + "',", '  ]']

    def test_list_wrap_for_width_rewrap_same_length(self) -> None:
        if False:
            print('Hello World!')
        long_a = 'a' * 30
        long_b = 'b' * 30
        long_c = 'c' * 30
        l1 = [long_a, long_b, long_c]
        l2 = [long_b, long_c, long_a]
        diff = callequal(l1, l2, verbose=True)
        assert diff == ["['aaaaaaaaaaa...cccccccccccc'] == ['bbbbbbbbbbb...aaaaaaaaaaaa']", "At index 0 diff: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' != 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'", 'Full diff:', '  [', "+  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',", "   'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',", "   'cccccccccccccccccccccccccccccc',", "-  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',", '  ]']

    def test_list_dont_wrap_strings(self) -> None:
        if False:
            i = 10
            return i + 15
        long_a = 'a' * 10
        l1 = ['a'] + [long_a for _ in range(0, 7)]
        l2 = ['should not get wrapped']
        diff = callequal(l1, l2, verbose=True)
        assert diff == ["['a', 'aaaaaa...aaaaaaa', ...] == ['should not get wrapped']", "At index 0 diff: 'a' != 'should not get wrapped'", "Left contains 7 more items, first extra item: 'aaaaaaaaaa'", 'Full diff:', '  [', "-  'should not get wrapped',", "+  'a',", "+  'aaaaaaaaaa',", "+  'aaaaaaaaaa',", "+  'aaaaaaaaaa',", "+  'aaaaaaaaaa',", "+  'aaaaaaaaaa',", "+  'aaaaaaaaaa',", "+  'aaaaaaaaaa',", '  ]']

    def test_dict_wrap(self) -> None:
        if False:
            i = 10
            return i + 15
        d1 = {'common': 1, 'env': {'env1': 1, 'env2': 2}}
        d2 = {'common': 1, 'env': {'env1': 1}}
        diff = callequal(d1, d2, verbose=True)
        assert diff == ["{'common': 1,...1, 'env2': 2}} == {'common': 1,...: {'env1': 1}}", 'Omitting 1 identical items, use -vv to show', 'Differing items:', "{'env': {'env1': 1, 'env2': 2}} != {'env': {'env1': 1}}", 'Full diff:', "- {'common': 1, 'env': {'env1': 1}}", "+ {'common': 1, 'env': {'env1': 1, 'env2': 2}}", '?                                +++++++++++']
        long_a = 'a' * 80
        sub = {'long_a': long_a, 'sub1': {'long_a': 'substring that gets wrapped ' * 2}}
        d1 = {'env': {'sub': sub}}
        d2 = {'env': {'sub': sub}, 'new': 1}
        diff = callequal(d1, d2, verbose=True)
        assert diff == ["{'env': {'sub... wrapped '}}}} == {'env': {'sub...}}}, 'new': 1}", 'Omitting 1 identical items, use -vv to show', 'Right contains 1 more item:', "{'new': 1}", 'Full diff:', '  {', "   'env': {'sub': {'long_a': '" + long_a + "',", "                   'sub1': {'long_a': 'substring that gets wrapped substring '", "                                      'that gets wrapped '}}},", "-  'new': 1,", '  }']

    def test_dict(self) -> None:
        if False:
            return 10
        expl = callequal({'a': 0}, {'a': 1})
        assert expl is not None
        assert len(expl) > 1

    def test_dict_omitting(self) -> None:
        if False:
            print('Hello World!')
        lines = callequal({'a': 0, 'b': 1}, {'a': 1, 'b': 1})
        assert lines is not None
        assert lines[1].startswith('Omitting 1 identical item')
        assert 'Common items' not in lines
        for line in lines[1:]:
            assert 'b' not in line

    def test_dict_omitting_with_verbosity_1(self) -> None:
        if False:
            print('Hello World!')
        'Ensure differing items are visible for verbosity=1 (#1512).'
        lines = callequal({'a': 0, 'b': 1}, {'a': 1, 'b': 1}, verbose=1)
        assert lines is not None
        assert lines[1].startswith('Omitting 1 identical item')
        assert lines[2].startswith('Differing items')
        assert lines[3] == "{'a': 0} != {'a': 1}"
        assert 'Common items' not in lines

    def test_dict_omitting_with_verbosity_2(self) -> None:
        if False:
            print('Hello World!')
        lines = callequal({'a': 0, 'b': 1}, {'a': 1, 'b': 1}, verbose=2)
        assert lines is not None
        assert lines[1].startswith('Common items:')
        assert 'Omitting' not in lines[1]
        assert lines[2] == "{'b': 1}"

    def test_dict_different_items(self) -> None:
        if False:
            while True:
                i = 10
        lines = callequal({'a': 0}, {'b': 1, 'c': 2}, verbose=2)
        assert lines == ["{'a': 0} == {'b': 1, 'c': 2}", 'Left contains 1 more item:', "{'a': 0}", 'Right contains 2 more items:', "{'b': 1, 'c': 2}", 'Full diff:', "- {'b': 1, 'c': 2}", "+ {'a': 0}"]
        lines = callequal({'b': 1, 'c': 2}, {'a': 0}, verbose=2)
        assert lines == ["{'b': 1, 'c': 2} == {'a': 0}", 'Left contains 2 more items:', "{'b': 1, 'c': 2}", 'Right contains 1 more item:', "{'a': 0}", 'Full diff:', "- {'a': 0}", "+ {'b': 1, 'c': 2}"]

    def test_sequence_different_items(self) -> None:
        if False:
            print('Hello World!')
        lines = callequal((1, 2), (3, 4, 5), verbose=2)
        assert lines == ['(1, 2) == (3, 4, 5)', 'At index 0 diff: 1 != 3', 'Right contains one more item: 5', 'Full diff:', '- (3, 4, 5)', '+ (1, 2)']
        lines = callequal((1, 2, 3), (4,), verbose=2)
        assert lines == ['(1, 2, 3) == (4,)', 'At index 0 diff: 1 != 4', 'Left contains 2 more items, first extra item: 2', 'Full diff:', '- (4,)', '+ (1, 2, 3)']

    def test_set(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = callequal({0, 1}, {0, 2})
        assert expl is not None
        assert len(expl) > 1

    def test_frozenzet(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = callequal(frozenset([0, 1]), {0, 2})
        assert expl is not None
        assert len(expl) > 1

    def test_Sequence(self) -> None:
        if False:
            i = 10
            return i + 15

        class TestSequence(MutableSequence[int]):

            def __init__(self, iterable):
                if False:
                    while True:
                        i = 10
                self.elements = list(iterable)

            def __getitem__(self, item):
                if False:
                    print('Hello World!')
                return self.elements[item]

            def __len__(self):
                if False:
                    i = 10
                    return i + 15
                return len(self.elements)

            def __setitem__(self, item, value):
                if False:
                    return 10
                pass

            def __delitem__(self, item):
                if False:
                    return 10
                pass

            def insert(self, item, index):
                if False:
                    return 10
                pass
        expl = callequal(TestSequence([0, 1]), list([0, 2]))
        assert expl is not None
        assert len(expl) > 1

    def test_list_tuples(self) -> None:
        if False:
            while True:
                i = 10
        expl = callequal([], [(1, 2)])
        assert expl is not None
        assert len(expl) > 1
        expl = callequal([(1, 2)], [])
        assert expl is not None
        assert len(expl) > 1

    def test_list_bad_repr(self) -> None:
        if False:
            while True:
                i = 10

        class A:

            def __repr__(self):
                if False:
                    return 10
                raise ValueError(42)
        expl = callequal([], [A()])
        assert expl is not None
        assert 'ValueError' in ''.join(expl)
        expl = callequal({}, {'1': A()}, verbose=2)
        assert expl is not None
        assert expl[0].startswith('{} == <[ValueError')
        assert 'raised in repr' in expl[0]
        assert expl[1:] == ['(pytest_assertion plugin: representation of details failed: {}:{}: ValueError: 42.'.format(__file__, A.__repr__.__code__.co_firstlineno + 1), ' Probably an object has a faulty __repr__.)']

    def test_one_repr_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The faulty empty string repr did trigger an unbound local error in _diff_text.'

        class A(str):

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return ''
        expl = callequal(A(), '')
        assert not expl

    def test_repr_no_exc(self) -> None:
        if False:
            i = 10
            return i + 15
        expl = callequal('foo', 'bar')
        assert expl is not None
        assert 'raised in repr()' not in ' '.join(expl)

    def test_unicode(self) -> None:
        if False:
            print('Hello World!')
        assert callequal('£€', '£') == ["'£€' == '£'", '- £', '+ £€']

    def test_nonascii_text(self) -> None:
        if False:
            print('Hello World!')
        '\n        :issue: 877\n        non ascii python2 str caused a UnicodeDecodeError\n        '

        class A(str):

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return 'ÿ'
        expl = callequal(A(), '1')
        assert expl == ["ÿ == '1'", '- 1']

    def test_format_nonascii_explanation(self) -> None:
        if False:
            return 10
        assert util.format_explanation('λ')

    def test_mojibake(self) -> None:
        if False:
            return 10
        left = b'e'
        right = b'\xc3\xa9'
        expl = callequal(left, right)
        assert expl is not None
        for line in expl:
            assert isinstance(line, str)
        msg = '\n'.join(expl)
        assert msg

    def test_nfc_nfd_same_string(self) -> None:
        if False:
            print('Hello World!')
        left = 'hyvä'
        right = 'hyvä'
        expl = callequal(left, right)
        assert expl == ["'hyv\\xe4' == 'hyva\\u0308'", f'- {str(right)}', f'+ {str(left)}']
        expl = callequal(left, right, verbose=2)
        assert expl == ["'hyv\\xe4' == 'hyva\\u0308'", f'- {str(right)}', f'+ {str(left)}']

class TestAssert_reprcompare_dataclass:

    def test_dataclasses(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p = pytester.copy_example('dataclasses/test_compare_dataclasses.py')
        result = pytester.runpytest(p)
        result.assert_outcomes(failed=1, passed=0)
        result.stdout.fnmatch_lines(['E         Omitting 1 identical items, use -vv to show', 'E         Differing attributes:', "E         ['field_b']", 'E         ', 'E         Drill down into differing attribute field_b:', "E           field_b: 'b' != 'c'", 'E           - c', 'E           + b'], consecutive=True)

    def test_recursive_dataclasses(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.copy_example('dataclasses/test_compare_recursive_dataclasses.py')
        result = pytester.runpytest(p)
        result.assert_outcomes(failed=1, passed=0)
        result.stdout.fnmatch_lines(['E         Omitting 1 identical items, use -vv to show', 'E         Differing attributes:', "E         ['g', 'h', 'j']", 'E         ', 'E         Drill down into differing attribute g:', "E           g: S(a=10, b='ten') != S(a=20, b='xxx')...", 'E         ', "E         ...Full output truncated (51 lines hidden), use '-vv' to show"], consecutive=True)

    def test_recursive_dataclasses_verbose(self, pytester: Pytester) -> None:
        if False:
            return 10
        p = pytester.copy_example('dataclasses/test_compare_recursive_dataclasses.py')
        result = pytester.runpytest(p, '-vv')
        result.assert_outcomes(failed=1, passed=0)
        result.stdout.fnmatch_lines(['E         Matching attributes:', "E         ['i']", 'E         Differing attributes:', "E         ['g', 'h', 'j']", 'E         ', 'E         Drill down into differing attribute g:', "E           g: S(a=10, b='ten') != S(a=20, b='xxx')", 'E           ', 'E           Differing attributes:', "E           ['a', 'b']", 'E           ', 'E           Drill down into differing attribute a:', 'E             a: 10 != 20', 'E           ', 'E           Drill down into differing attribute b:', "E             b: 'ten' != 'xxx'", 'E             - xxx', 'E             + ten', 'E         ', 'E         Drill down into differing attribute h:'], consecutive=True)

    def test_dataclasses_verbose(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.copy_example('dataclasses/test_compare_dataclasses_verbose.py')
        result = pytester.runpytest(p, '-vv')
        result.assert_outcomes(failed=1, passed=0)
        result.stdout.fnmatch_lines(['*Matching attributes:*', "*['field_a']*", '*Differing attributes:*', "*field_b: 'b' != 'c'*"])

    def test_dataclasses_with_attribute_comparison_off(self, pytester: Pytester) -> None:
        if False:
            return 10
        p = pytester.copy_example('dataclasses/test_compare_dataclasses_field_comparison_off.py')
        result = pytester.runpytest(p, '-vv')
        result.assert_outcomes(failed=0, passed=1)

    def test_comparing_two_different_data_classes(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.copy_example('dataclasses/test_compare_two_different_dataclasses.py')
        result = pytester.runpytest(p, '-vv')
        result.assert_outcomes(failed=0, passed=1)

    def test_data_classes_with_custom_eq(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p = pytester.copy_example('dataclasses/test_compare_dataclasses_with_custom_eq.py')
        result = pytester.runpytest(p, '-vv')
        result.assert_outcomes(failed=1, passed=0)
        result.stdout.no_re_match_line('.*Differing attributes.*')

    def test_data_classes_with_initvar(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.copy_example('dataclasses/test_compare_initvar.py')
        result = pytester.runpytest(p, '-vv')
        result.assert_outcomes(failed=1, passed=0)
        result.stdout.no_re_match_line('.*AttributeError.*')

class TestAssert_reprcompare_attrsclass:

    def test_attrs(self) -> None:
        if False:
            while True:
                i = 10

        @attr.s
        class SimpleDataObject:
            field_a = attr.ib()
            field_b = attr.ib()
        left = SimpleDataObject(1, 'b')
        right = SimpleDataObject(1, 'c')
        lines = callequal(left, right)
        assert lines is not None
        assert lines[2].startswith('Omitting 1 identical item')
        assert 'Matching attributes' not in lines
        for line in lines[2:]:
            assert 'field_a' not in line

    def test_attrs_recursive(self) -> None:
        if False:
            print('Hello World!')

        @attr.s
        class OtherDataObject:
            field_c = attr.ib()
            field_d = attr.ib()

        @attr.s
        class SimpleDataObject:
            field_a = attr.ib()
            field_b = attr.ib()
        left = SimpleDataObject(OtherDataObject(1, 'a'), 'b')
        right = SimpleDataObject(OtherDataObject(1, 'b'), 'b')
        lines = callequal(left, right)
        assert lines is not None
        assert 'Matching attributes' not in lines
        for line in lines[1:]:
            assert 'field_b:' not in line
            assert 'field_c:' not in line

    def test_attrs_recursive_verbose(self) -> None:
        if False:
            return 10

        @attr.s
        class OtherDataObject:
            field_c = attr.ib()
            field_d = attr.ib()

        @attr.s
        class SimpleDataObject:
            field_a = attr.ib()
            field_b = attr.ib()
        left = SimpleDataObject(OtherDataObject(1, 'a'), 'b')
        right = SimpleDataObject(OtherDataObject(1, 'b'), 'b')
        lines = callequal(left, right)
        assert lines is not None
        assert "    field_d: 'a' != 'b'" in lines

    def test_attrs_verbose(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        @attr.s
        class SimpleDataObject:
            field_a = attr.ib()
            field_b = attr.ib()
        left = SimpleDataObject(1, 'b')
        right = SimpleDataObject(1, 'c')
        lines = callequal(left, right, verbose=2)
        assert lines is not None
        assert lines[2].startswith('Matching attributes:')
        assert 'Omitting' not in lines[2]
        assert lines[3] == "['field_a']"

    def test_attrs_with_attribute_comparison_off(self) -> None:
        if False:
            i = 10
            return i + 15

        @attr.s
        class SimpleDataObject:
            field_a = attr.ib()
            field_b = attr.ib(eq=False)
        left = SimpleDataObject(1, 'b')
        right = SimpleDataObject(1, 'b')
        lines = callequal(left, right, verbose=2)
        assert lines is not None
        assert lines[2].startswith('Matching attributes:')
        assert 'Omitting' not in lines[1]
        assert lines[3] == "['field_a']"
        for line in lines[3:]:
            assert 'field_b' not in line

    def test_comparing_two_different_attrs_classes(self) -> None:
        if False:
            return 10

        @attr.s
        class SimpleDataObjectOne:
            field_a = attr.ib()
            field_b = attr.ib()

        @attr.s
        class SimpleDataObjectTwo:
            field_a = attr.ib()
            field_b = attr.ib()
        left = SimpleDataObjectOne(1, 'b')
        right = SimpleDataObjectTwo(1, 'c')
        lines = callequal(left, right)
        assert lines is None

    def test_attrs_with_auto_detect_and_custom_eq(self) -> None:
        if False:
            print('Hello World!')

        @attr.s(auto_detect=True)
        class SimpleDataObject:
            field_a = attr.ib()

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                return super().__eq__(other)
        left = SimpleDataObject(1)
        right = SimpleDataObject(2)
        lines = callequal(left, right, verbose=2)
        assert lines is None

    def test_attrs_with_custom_eq(self) -> None:
        if False:
            return 10

        @attr.define(slots=False)
        class SimpleDataObject:
            field_a = attr.ib()

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                return super().__eq__(other)
        left = SimpleDataObject(1)
        right = SimpleDataObject(2)
        lines = callequal(left, right, verbose=2)
        assert lines is None

class TestAssert_reprcompare_namedtuple:

    def test_namedtuple(self) -> None:
        if False:
            print('Hello World!')
        NT = collections.namedtuple('NT', ['a', 'b'])
        left = NT(1, 'b')
        right = NT(1, 'c')
        lines = callequal(left, right)
        assert lines == ["NT(a=1, b='b') == NT(a=1, b='c')", '', 'Omitting 1 identical items, use -vv to show', 'Differing attributes:', "['b']", '', 'Drill down into differing attribute b:', "  b: 'b' != 'c'", '  - c', '  + b', 'Use -v to get more diff']

    def test_comparing_two_different_namedtuple(self) -> None:
        if False:
            print('Hello World!')
        NT1 = collections.namedtuple('NT1', ['a', 'b'])
        NT2 = collections.namedtuple('NT2', ['a', 'b'])
        left = NT1(1, 'b')
        right = NT2(2, 'b')
        lines = callequal(left, right)
        assert lines == ["NT1(a=1, b='b') == NT2(a=2, b='b')", 'At index 0 diff: 1 != 2', 'Use -v to get more diff']

class TestFormatExplanation:

    def test_special_chars_full(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile("\n            def test_foo():\n                assert '\\n}' == ''\n        ")
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*AssertionError*'])

    def test_fmt_simple(self) -> None:
        if False:
            i = 10
            return i + 15
        expl = 'assert foo'
        assert util.format_explanation(expl) == 'assert foo'

    def test_fmt_where(self) -> None:
        if False:
            return 10
        expl = '\n'.join(['assert 1', '{1 = foo', '} == 2'])
        res = '\n'.join(['assert 1 == 2', ' +  where 1 = foo'])
        assert util.format_explanation(expl) == res

    def test_fmt_and(self) -> None:
        if False:
            while True:
                i = 10
        expl = '\n'.join(['assert 1', '{1 = foo', '} == 2', '{2 = bar', '}'])
        res = '\n'.join(['assert 1 == 2', ' +  where 1 = foo', ' +  and   2 = bar'])
        assert util.format_explanation(expl) == res

    def test_fmt_where_nested(self) -> None:
        if False:
            i = 10
            return i + 15
        expl = '\n'.join(['assert 1', '{1 = foo', '{foo = bar', '}', '} == 2'])
        res = '\n'.join(['assert 1 == 2', ' +  where 1 = foo', ' +    where foo = bar'])
        assert util.format_explanation(expl) == res

    def test_fmt_newline(self) -> None:
        if False:
            i = 10
            return i + 15
        expl = '\n'.join(['assert "foo" == "bar"', '~- foo', '~+ bar'])
        res = '\n'.join(['assert "foo" == "bar"', '  - foo', '  + bar'])
        assert util.format_explanation(expl) == res

    def test_fmt_newline_escaped(self) -> None:
        if False:
            print('Hello World!')
        expl = '\n'.join(['assert foo == bar', 'baz'])
        res = 'assert foo == bar\\nbaz'
        assert util.format_explanation(expl) == res

    def test_fmt_newline_before_where(self) -> None:
        if False:
            i = 10
            return i + 15
        expl = '\n'.join(['the assertion message here', '>assert 1', '{1 = foo', '} == 2', '{2 = bar', '}'])
        res = '\n'.join(['the assertion message here', 'assert 1 == 2', ' +  where 1 = foo', ' +  and   2 = bar'])
        assert util.format_explanation(expl) == res

    def test_fmt_multi_newline_before_where(self) -> None:
        if False:
            while True:
                i = 10
        expl = '\n'.join(['the assertion', '~message here', '>assert 1', '{1 = foo', '} == 2', '{2 = bar', '}'])
        res = '\n'.join(['the assertion', '  message here', 'assert 1 == 2', ' +  where 1 = foo', ' +  and   2 = bar'])
        assert util.format_explanation(expl) == res

class TestTruncateExplanation:
    LINES_IN_TRUNCATION_MSG = 2

    def test_doesnt_truncate_when_input_is_empty_list(self) -> None:
        if False:
            return 10
        expl: List[str] = []
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=100)
        assert result == expl

    def test_doesnt_truncate_at_when_input_is_5_lines_and_LT_max_chars(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = ['a' * 100 for x in range(5)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=8 * 80)
        assert result == expl

    def test_truncates_at_8_lines_when_given_list_of_empty_strings(self) -> None:
        if False:
            return 10
        expl = ['' for x in range(50)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=100)
        assert len(result) != len(expl)
        assert result != expl
        assert len(result) == 8 + self.LINES_IN_TRUNCATION_MSG
        assert 'Full output truncated' in result[-1]
        assert '42 lines hidden' in result[-1]
        last_line_before_trunc_msg = result[-self.LINES_IN_TRUNCATION_MSG - 1]
        assert last_line_before_trunc_msg.endswith('...')

    def test_truncates_at_8_lines_when_first_8_lines_are_LT_max_chars(self) -> None:
        if False:
            i = 10
            return i + 15
        total_lines = 100
        expl = ['a' for x in range(total_lines)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=8 * 80)
        assert result != expl
        assert len(result) == 8 + self.LINES_IN_TRUNCATION_MSG
        assert 'Full output truncated' in result[-1]
        assert f'{total_lines - 8} lines hidden' in result[-1]
        last_line_before_trunc_msg = result[-self.LINES_IN_TRUNCATION_MSG - 1]
        assert last_line_before_trunc_msg.endswith('...')

    def test_truncates_at_8_lines_when_there_is_one_line_to_remove(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The number of line in the result is 9, the same number as if we truncated.'
        expl = ['a' for x in range(9)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=8 * 80)
        assert result == expl
        assert 'truncated' not in result[-1]

    def test_truncates_edgecase_when_truncation_message_makes_the_result_longer_for_chars(self) -> None:
        if False:
            return 10
        line = 'a' * 10
        expl = [line, line]
        result = truncate._truncate_explanation(expl, max_lines=10, max_chars=10)
        assert result == [line, line]

    def test_truncates_edgecase_when_truncation_message_makes_the_result_longer_for_lines(self) -> None:
        if False:
            print('Hello World!')
        line = 'a' * 10
        expl = [line, line]
        result = truncate._truncate_explanation(expl, max_lines=1, max_chars=100)
        assert result == [line, line]

    def test_truncates_at_8_lines_when_first_8_lines_are_EQ_max_chars(self) -> None:
        if False:
            print('Hello World!')
        expl = [chr(97 + x) * 80 for x in range(16)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=8 * 80)
        assert result != expl
        assert len(result) == 16 - 8 + self.LINES_IN_TRUNCATION_MSG
        assert 'Full output truncated' in result[-1]
        assert '8 lines hidden' in result[-1]
        last_line_before_trunc_msg = result[-self.LINES_IN_TRUNCATION_MSG - 1]
        assert last_line_before_trunc_msg.endswith('...')

    def test_truncates_at_4_lines_when_first_4_lines_are_GT_max_chars(self) -> None:
        if False:
            while True:
                i = 10
        expl = ['a' * 250 for x in range(10)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=999)
        assert result != expl
        assert len(result) == 4 + self.LINES_IN_TRUNCATION_MSG
        assert 'Full output truncated' in result[-1]
        assert '7 lines hidden' in result[-1]
        last_line_before_trunc_msg = result[-self.LINES_IN_TRUNCATION_MSG - 1]
        assert last_line_before_trunc_msg.endswith('...')

    def test_truncates_at_1_line_when_first_line_is_GT_max_chars(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expl = ['a' * 250 for x in range(1000)]
        result = truncate._truncate_explanation(expl, max_lines=8, max_chars=100)
        assert result != expl
        assert len(result) == 1 + self.LINES_IN_TRUNCATION_MSG
        assert 'Full output truncated' in result[-1]
        assert '1000 lines hidden' in result[-1]
        last_line_before_trunc_msg = result[-self.LINES_IN_TRUNCATION_MSG - 1]
        assert last_line_before_trunc_msg.endswith('...')

    def test_full_output_truncated(self, monkeypatch, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test against full runpytest() output.'
        line_count = 7
        line_len = 100
        expected_truncated_lines = 1
        pytester.makepyfile("\n            def test_many_lines():\n                a = list([str(i)[0] * %d for i in range(%d)])\n                b = a[::2]\n                a = '\\n'.join(map(str, a))\n                b = '\\n'.join(map(str, b))\n                assert a == b\n        " % (line_len, line_count))
        monkeypatch.delenv('CI', raising=False)
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*+ 1*', '*+ 3*', '*+ 5*', '*truncated (%d line hidden)*use*-vv*' % expected_truncated_lines])
        result = pytester.runpytest('-vv')
        result.stdout.fnmatch_lines(['* 6*'])
        monkeypatch.setenv('CI', '1')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 6*'])

def test_python25_compile_issue257(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        def test_rewritten():\n            assert 1 == 2\n        # some comment\n    ')
    result = pytester.runpytest()
    assert result.ret == 1
    result.stdout.fnmatch_lines('\n            *E*assert 1 == 2*\n            *1 failed*\n    ')

def test_rewritten(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def test_rewritten():\n            assert "@py_builtins" in globals()\n    ')
    assert pytester.runpytest().ret == 0

def test_reprcompare_notin() -> None:
    if False:
        while True:
            i = 10
    assert callop('not in', 'foo', 'aaafoobbb') == ["'foo' not in 'aaafoobbb'", "'foo' is contained here:", '  aaafoobbb', '?    +++']

def test_reprcompare_whitespaces() -> None:
    if False:
        i = 10
        return i + 15
    assert callequal('\r\n', '\n') == ["'\\r\\n' == '\\n'", 'Strings contain only whitespace, escaping them using repr()', "- '\\n'", "+ '\\r\\n'", '?  ++']

class TestSetAssertions:

    @pytest.mark.parametrize('op', ['>=', '>', '<=', '<', '=='])
    def test_set_extra_item(self, op, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(f'\n            def test_hello():\n                x = set("hello x")\n                y = set("hello y")\n                assert x {op} y\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*def test_hello():*', f'*assert x {op} y*'])
        if op in ['>=', '>', '==']:
            result.stdout.fnmatch_lines(['*E*Extra items in the right set:*', "*E*'y'"])
        if op in ['<=', '<', '==']:
            result.stdout.fnmatch_lines(['*E*Extra items in the left set:*', "*E*'x'"])

    @pytest.mark.parametrize('op', ['>', '<', '!='])
    def test_set_proper_superset_equal(self, pytester: Pytester, op) -> None:
        if False:
            return 10
        pytester.makepyfile(f'\n            def test_hello():\n                x = set([1, 2, 3])\n                y = x.copy()\n                assert x {op} y\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*def test_hello():*', f'*assert x {op} y*', '*E*Both sets are equal*'])

    def test_pytest_assertrepr_compare_integration(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_hello():\n                x = set(range(100))\n                y = x.copy()\n                y.remove(50)\n                assert x == y\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*def test_hello():*', '*assert x == y*', '*E*Extra items*left*', '*E*50*', '*= 1 failed in*'])

def test_assertrepr_loaded_per_dir(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile(test_base=['def test_base(): assert 1 == 2'])
    a = pytester.mkdir('a')
    a.joinpath('test_a.py').write_text('def test_a(): assert 1 == 2', encoding='utf-8')
    a.joinpath('conftest.py').write_text('def pytest_assertrepr_compare(): return ["summary a"]', encoding='utf-8')
    b = pytester.mkdir('b')
    b.joinpath('test_b.py').write_text('def test_b(): assert 1 == 2', encoding='utf-8')
    b.joinpath('conftest.py').write_text('def pytest_assertrepr_compare(): return ["summary b"]', encoding='utf-8')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*def test_base():*', '*E*assert 1 == 2*', '*def test_a():*', '*E*assert summary a*', '*def test_b():*', '*E*assert summary b*'])

def test_assertion_options(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        def test_hello():\n            x = 3\n            assert x == 4\n    ')
    result = pytester.runpytest()
    assert '3 == 4' in result.stdout.str()
    result = pytester.runpytest_subprocess('--assert=plain')
    result.stdout.no_fnmatch_line('*3 == 4*')

def test_triple_quoted_string_issue113(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        def test_hello():\n            assert "" == \'\'\'\n    \'\'\'')
    result = pytester.runpytest('--fulltrace')
    result.stdout.fnmatch_lines(['*1 failed*'])
    result.stdout.no_fnmatch_line('*SyntaxError*')

def test_traceback_failure(pytester: Pytester) -> None:
    if False:
        return 10
    p1 = pytester.makepyfile('\n        def g():\n            return 2\n        def f(x):\n            assert x == g()\n        def test_onefails():\n            f(3)\n    ')
    result = pytester.runpytest(p1, '--tb=long')
    result.stdout.fnmatch_lines(['*test_traceback_failure.py F*', '====* FAILURES *====', '____*____', '', '    def test_onefails():', '>       f(3)', '', '*test_*.py:6: ', '_ _ _ *', '    def f(x):', '>       assert x == g()', 'E       assert 3 == 2', 'E        +  where 2 = g()', '', '*test_traceback_failure.py:4: AssertionError'])
    result = pytester.runpytest(p1)
    result.stdout.fnmatch_lines(['*test_traceback_failure.py F*', '====* FAILURES *====', '____*____', '', '    def test_onefails():', '>       f(3)', '', '*test_*.py:6: ', '', '    def f(x):', '>       assert x == g()', 'E       assert 3 == 2', 'E        +  where 2 = g()', '', '*test_traceback_failure.py:4: AssertionError'])

def test_exception_handling_no_traceback(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Handle chain exceptions in tasks submitted by the multiprocess module (#1984).'
    p1 = pytester.makepyfile('\n        from multiprocessing import Pool\n\n        def process_task(n):\n            assert n == 10\n\n        def multitask_job():\n            tasks = [1]\n            with Pool(processes=1) as pool:\n                pool.map(process_task, tasks)\n\n        def test_multitask_job():\n            multitask_job()\n    ')
    pytester.syspathinsert()
    result = pytester.runpytest(p1, '--tb=long')
    result.stdout.fnmatch_lines(['====* FAILURES *====', '*multiprocessing.pool.RemoteTraceback:*', 'Traceback (most recent call last):', '*assert n == 10', 'The above exception was the direct cause of the following exception:', '> * multitask_job()'])

@pytest.mark.skipif("'__pypy__' in sys.builtin_module_names")
@pytest.mark.parametrize('cmdline_args, warning_output', [(['-OO', '-m', 'pytest', '-h'], ['warning :*PytestConfigWarning:*assert statements are not executed*']), (['-OO', '-m', 'pytest'], ['=*= warnings summary =*=', '*PytestConfigWarning:*assert statements are not executed*']), (['-OO', '-m', 'pytest', '--assert=plain'], ['=*= warnings summary =*=', '*PytestConfigWarning: ASSERTIONS ARE NOT EXECUTED and FAILING TESTS WILL PASS.  Are you using python -O?'])])
def test_warn_missing(pytester: Pytester, cmdline_args, warning_output) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('')
    result = pytester.run(sys.executable, *cmdline_args)
    result.stdout.fnmatch_lines(warning_output)

def test_recursion_source_decode(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def test_something():\n            pass\n    ')
    pytester.makeini('\n        [pytest]\n        python_files = *.py\n    ')
    result = pytester.runpytest('--collect-only')
    result.stdout.fnmatch_lines('\n        <Module*>\n    ')

def test_AssertionError_message(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        def test_hello():\n            x,y = 1,2\n            assert 0, (x,y)\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines('\n        *def test_hello*\n        *assert 0, (x,y)*\n        *AssertionError: (1, 2)*\n    ')

def test_diff_newline_at_end(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile("\n        def test_diff():\n            assert 'asdf' == 'asdf\\n'\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines("\n        *assert 'asdf' == 'asdf\\n'\n        *  - asdf\n        *  ?     -\n        *  + asdf\n    ")

@pytest.mark.filterwarnings('default')
def test_assert_tuple_warning(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    msg = 'assertion is always true'
    pytester.makepyfile("\n        def test_tuple():\n            assert(False, 'you shall not pass')\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines([f'*test_assert_tuple_warning.py:2:*{msg}*'])
    pytester.makepyfile('\n        def test_tuple():\n            assert ()\n    ')
    result = pytester.runpytest()
    assert msg not in result.stdout.str()

def test_assert_indirect_tuple_no_warning(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile("\n        def test_tuple():\n            tpl = ('foo', 'bar')\n            assert tpl\n    ")
    result = pytester.runpytest()
    output = '\n'.join(result.stdout.lines)
    assert 'WR1' not in output

def test_assert_with_unicode(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile("        def test_unicode():\n            assert '유니코드' == 'Unicode'\n        ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*AssertionError*'])

def test_raise_unprintable_assertion_error(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile("\n        def test_raise_assertion_error():\n            raise AssertionError('\\xff')\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines([">       raise AssertionError('\\xff')", 'E       AssertionError: *'])

def test_raise_assertion_error_raising_repr(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        class RaisingRepr(object):\n            def __repr__(self):\n                raise Exception()\n        def test_raising_repr():\n            raise AssertionError(RaisingRepr())\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['E       AssertionError: <exception str() failed>'])

def test_issue_1944(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def f():\n            return\n\n        assert f() == 10\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*1 error*'])
    assert "AttributeError: 'Module' object has no attribute '_obj'" not in result.stdout.str()

def test_exit_from_assertrepr_compare(monkeypatch) -> None:
    if False:
        while True:
            i = 10

    def raise_exit(obj):
        if False:
            return 10
        outcomes.exit('Quitting debugger')
    monkeypatch.setattr(util, 'istext', raise_exit)
    with pytest.raises(outcomes.Exit, match='Quitting debugger'):
        callequal(1, 1)

def test_assertion_location_with_coverage(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'This used to report the wrong location when run with coverage (#5754).'
    p = pytester.makepyfile('\n        def test():\n            assert False, 1\n            assert False, 2\n        ')
    result = pytester.runpytest(str(p))
    result.stdout.fnmatch_lines(['>       assert False, 1', 'E       AssertionError: 1', 'E       assert False', '*= 1 failed in*'])

def test_reprcompare_verbose_long() -> None:
    if False:
        print('Hello World!')
    a = {f'v{i}': i for i in range(11)}
    b = a.copy()
    b['v2'] += 10
    lines = callop('==', a, b, verbose=2)
    assert lines is not None
    assert lines[0] == "{'v0': 0, 'v1': 1, 'v2': 2, 'v3': 3, 'v4': 4, 'v5': 5, 'v6': 6, 'v7': 7, 'v8': 8, 'v9': 9, 'v10': 10} == {'v0': 0, 'v1': 1, 'v2': 12, 'v3': 3, 'v4': 4, 'v5': 5, 'v6': 6, 'v7': 7, 'v8': 8, 'v9': 9, 'v10': 10}"

@pytest.mark.parametrize('enable_colors', [True, False])
@pytest.mark.parametrize(('test_code', 'expected_lines'), (('\n            def test():\n                assert [0, 1] == [0, 2]\n            ', ['{bold}{red}E         {light-red}- [0, 2]{hl-reset}{endline}{reset}', '{bold}{red}E         {light-green}+ [0, 1]{hl-reset}{endline}{reset}']), ('\n            def test():\n                assert {f"number-is-{i}": i for i in range(1, 6)} == {\n                    f"number-is-{i}": i for i in range(5)\n                }\n            ', ['{bold}{red}E         {light-gray} {hl-reset} {{{endline}{reset}', "{bold}{red}E         {light-gray} {hl-reset}  'number-is-1': 1,{endline}{reset}", "{bold}{red}E         {light-green}+  'number-is-5': 5,{hl-reset}{endline}{reset}"])))
def test_comparisons_handle_colors(pytester: Pytester, color_mapping, enable_colors, test_code, expected_lines) -> None:
    if False:
        return 10
    p = pytester.makepyfile(test_code)
    result = pytester.runpytest(f"--color={('yes' if enable_colors else 'no')}", '-vv', str(p))
    formatter = color_mapping.format_for_fnmatch if enable_colors else color_mapping.strip_colors
    result.stdout.fnmatch_lines(formatter(expected_lines), consecutive=False)