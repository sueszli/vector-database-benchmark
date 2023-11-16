from _pytest.pytester import Pytester

def test_no_items_should_not_show_output(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = pytester.runpytest('--fixtures-per-test')
    result.stdout.no_fnmatch_line('*fixtures used by*')
    assert result.ret == 0

def test_fixtures_in_module(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def _arg0():\n            """hidden arg0 fixture"""\n        @pytest.fixture\n        def arg1():\n            """arg1 docstring"""\n        def test_arg1(arg1):\n            pass\n    ')
    result = pytester.runpytest('--fixtures-per-test', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*fixtures used by test_arg1*', '*(test_fixtures_in_module.py:9)*', 'arg1 -- test_fixtures_in_module.py:6', '    arg1 docstring'])
    result.stdout.no_fnmatch_line('*_arg0*')

def test_fixtures_in_conftest(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makeconftest('\n        import pytest\n        @pytest.fixture\n        def arg1():\n            """arg1 docstring"""\n        @pytest.fixture\n        def arg2():\n            """arg2 docstring"""\n        @pytest.fixture\n        def arg3(arg1, arg2):\n            """arg3\n            docstring\n            """\n    ')
    p = pytester.makepyfile('\n        def test_arg2(arg2):\n            pass\n        def test_arg3(arg3):\n            pass\n    ')
    result = pytester.runpytest('--fixtures-per-test', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*fixtures used by test_arg2*', '*(test_fixtures_in_conftest.py:2)*', 'arg2 -- conftest.py:6', '    arg2 docstring', '*fixtures used by test_arg3*', '*(test_fixtures_in_conftest.py:4)*', 'arg1 -- conftest.py:3', '    arg1 docstring', 'arg2 -- conftest.py:6', '    arg2 docstring', 'arg3 -- conftest.py:9', '    arg3'])

def test_should_show_fixtures_used_by_test(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makeconftest('\n        import pytest\n        @pytest.fixture\n        def arg1():\n            """arg1 from conftest"""\n        @pytest.fixture\n        def arg2():\n            """arg2 from conftest"""\n    ')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg1():\n            """arg1 from testmodule"""\n        def test_args(arg1, arg2):\n            pass\n    ')
    result = pytester.runpytest('--fixtures-per-test', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*fixtures used by test_args*', '*(test_should_show_fixtures_used_by_test.py:6)*', 'arg1 -- test_should_show_fixtures_used_by_test.py:3', '    arg1 from testmodule', 'arg2 -- conftest.py:6', '    arg2 from conftest'])

def test_verbose_include_private_fixtures_and_loc(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makeconftest('\n        import pytest\n        @pytest.fixture\n        def _arg1():\n            """_arg1 from conftest"""\n        @pytest.fixture\n        def arg2(_arg1):\n            """arg2 from conftest"""\n    ')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg3():\n            """arg3 from testmodule"""\n        def test_args(arg2, arg3):\n            pass\n    ')
    result = pytester.runpytest('--fixtures-per-test', '-v', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*fixtures used by test_args*', '*(test_verbose_include_private_fixtures_and_loc.py:6)*', '_arg1 -- conftest.py:3', '    _arg1 from conftest', 'arg2 -- conftest.py:6', '    arg2 from conftest', 'arg3 -- test_verbose_include_private_fixtures_and_loc.py:3', '    arg3 from testmodule'])

def test_doctest_items(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        def foo():\n            """\n            >>> 1 + 1\n            2\n            """\n    ')
    pytester.maketxtfile('\n        >>> 1 + 1\n        2\n    ')
    result = pytester.runpytest('--fixtures-per-test', '--doctest-modules', '--doctest-glob=*.txt', '-v')
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*collected 2 items*'])

def test_multiline_docstring_in_module(pytester: Pytester) -> None:
    if False:
        return 10
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg1():\n            """Docstring content that spans across multiple lines,\n            through second line,\n            and through third line.\n\n            Docstring content that extends into a second paragraph.\n\n            Docstring content that extends into a third paragraph.\n            """\n        def test_arg1(arg1):\n            pass\n    ')
    result = pytester.runpytest('--fixtures-per-test', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*fixtures used by test_arg1*', '*(test_multiline_docstring_in_module.py:13)*', 'arg1 -- test_multiline_docstring_in_module.py:3', '    Docstring content that spans across multiple lines,', '    through second line,', '    and through third line.'])

def test_verbose_include_multiline_docstring(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg1():\n            """Docstring content that spans across multiple lines,\n            through second line,\n            and through third line.\n\n            Docstring content that extends into a second paragraph.\n\n            Docstring content that extends into a third paragraph.\n            """\n        def test_arg1(arg1):\n            pass\n    ')
    result = pytester.runpytest('--fixtures-per-test', '-v', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*fixtures used by test_arg1*', '*(test_verbose_include_multiline_docstring.py:13)*', 'arg1 -- test_verbose_include_multiline_docstring.py:3', '    Docstring content that spans across multiple lines,', '    through second line,', '    and through third line.', '    ', '    Docstring content that extends into a second paragraph.', '    ', '    Docstring content that extends into a third paragraph.'])