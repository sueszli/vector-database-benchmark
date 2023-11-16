import sys
from textwrap import dedent
from typing import Generator
from typing import List
from typing import Optional
import pytest
from _pytest.pytester import Pytester

@pytest.fixture()
def file_structure(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile(test_foo='\n        from foo import foo\n\n        def test_foo():\n            assert foo() == 1\n        ')
    pytester.makepyfile(test_bar='\n        from bar import bar\n\n        def test_bar():\n            assert bar() == 2\n        ')
    foo_py = pytester.mkdir('sub') / 'foo.py'
    content = dedent('\n        def foo():\n            return 1\n        ')
    foo_py.write_text(content, encoding='utf-8')
    bar_py = pytester.mkdir('sub2') / 'bar.py'
    content = dedent('\n        def bar():\n            return 2\n        ')
    bar_py.write_text(content, encoding='utf-8')

def test_one_dir(pytester: Pytester, file_structure) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makefile('.ini', pytest='[pytest]\npythonpath=sub\n')
    result = pytester.runpytest('test_foo.py')
    assert result.ret == 0
    result.assert_outcomes(passed=1)

def test_two_dirs(pytester: Pytester, file_structure) -> None:
    if False:
        print('Hello World!')
    pytester.makefile('.ini', pytest='[pytest]\npythonpath=sub sub2\n')
    result = pytester.runpytest('test_foo.py', 'test_bar.py')
    assert result.ret == 0
    result.assert_outcomes(passed=2)

def test_module_not_found(pytester: Pytester, file_structure) -> None:
    if False:
        print('Hello World!')
    'Without the pythonpath setting, the module should not be found.'
    pytester.makefile('.ini', pytest='[pytest]\n')
    result = pytester.runpytest('test_foo.py')
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    expected_error = "E   ModuleNotFoundError: No module named 'foo'"
    result.stdout.fnmatch_lines([expected_error])

def test_no_ini(pytester: Pytester, file_structure) -> None:
    if False:
        while True:
            i = 10
    'If no ini file, test should error.'
    result = pytester.runpytest('test_foo.py')
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    expected_error = "E   ModuleNotFoundError: No module named 'foo'"
    result.stdout.fnmatch_lines([expected_error])

def test_clean_up(pytester: Pytester) -> None:
    if False:
        return 10
    'Test that the plugin cleans up after itself.'
    pytester.makefile('.ini', pytest='[pytest]\npythonpath=I_SHALL_BE_REMOVED\n')
    pytester.makepyfile(test_foo='def test_foo(): pass')
    before: Optional[List[str]] = None
    after: Optional[List[str]] = None

    class Plugin:

        @pytest.hookimpl(wrapper=True, tryfirst=True)
        def pytest_unconfigure(self) -> Generator[None, None, None]:
            if False:
                print('Hello World!')
            nonlocal before, after
            before = sys.path.copy()
            try:
                return (yield)
            finally:
                after = sys.path.copy()
    result = pytester.runpytest_inprocess(plugins=[Plugin()])
    assert result.ret == 0
    assert before is not None
    assert after is not None
    assert any(('I_SHALL_BE_REMOVED' in entry for entry in before))
    assert not any(('I_SHALL_BE_REMOVED' in entry for entry in after))