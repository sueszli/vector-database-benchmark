from pathlib import Path
import pytest
from _pytest.cacheprovider import Cache
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.stepwise import STEPWISE_CACHE_DIR

@pytest.fixture
def stepwise_pytester(pytester: Pytester) -> Pytester:
    if False:
        while True:
            i = 10
    pytester.makeconftest("\ndef pytest_addoption(parser):\n    group = parser.getgroup('general')\n    group.addoption('--fail', action='store_true', dest='fail')\n    group.addoption('--fail-last', action='store_true', dest='fail_last')\n")
    pytester.makepyfile(test_a="\ndef test_success_before_fail():\n    assert 1\n\ndef test_fail_on_flag(request):\n    assert not request.config.getvalue('fail')\n\ndef test_success_after_fail():\n    assert 1\n\ndef test_fail_last_on_flag(request):\n    assert not request.config.getvalue('fail_last')\n\ndef test_success_after_last_fail():\n    assert 1\n")
    pytester.makepyfile(test_b='\ndef test_success():\n    assert 1\n')
    pytester.makeini('\n        [pytest]\n        cache_dir = .cache\n    ')
    return pytester

@pytest.fixture
def error_pytester(pytester: Pytester) -> Pytester:
    if False:
        return 10
    pytester.makepyfile(test_a='\ndef test_error(nonexisting_fixture):\n    assert 1\n\ndef test_success_after_fail():\n    assert 1\n')
    return pytester

@pytest.fixture
def broken_pytester(pytester: Pytester) -> Pytester:
    if False:
        return 10
    pytester.makepyfile(working_testfile='def test_proper(): assert 1', broken_testfile='foobar')
    return pytester

def _strip_resource_warnings(lines):
    if False:
        i = 10
        return i + 15
    return [x for x in lines if not x.startswith(('Exception ignored in:', 'ResourceWarning'))]

def test_run_without_stepwise(stepwise_pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    result = stepwise_pytester.runpytest('-v', '--strict-markers', '--fail')
    result.stdout.fnmatch_lines(['*test_success_before_fail PASSED*'])
    result.stdout.fnmatch_lines(['*test_fail_on_flag FAILED*'])
    result.stdout.fnmatch_lines(['*test_success_after_fail PASSED*'])

def test_stepwise_output_summary(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        import pytest\n        @pytest.mark.parametrize("expected", [True, True, True, True, False])\n        def test_data(expected):\n            assert expected\n        ')
    result = pytester.runpytest('-v', '--stepwise')
    result.stdout.fnmatch_lines(['stepwise: no previously failed tests, not skipping.'])
    result = pytester.runpytest('-v', '--stepwise')
    result.stdout.fnmatch_lines(['stepwise: skipping 4 already passed items.', '*1 failed, 4 deselected*'])

def test_fail_and_continue_with_stepwise(stepwise_pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = stepwise_pytester.runpytest('-v', '--strict-markers', '--stepwise', '--fail')
    assert _strip_resource_warnings(result.stderr.lines) == []
    stdout = result.stdout.str()
    assert 'test_success_before_fail PASSED' in stdout
    assert 'test_fail_on_flag FAILED' in stdout
    assert 'test_success_after_fail' not in stdout
    result = stepwise_pytester.runpytest('-v', '--strict-markers', '--stepwise')
    assert _strip_resource_warnings(result.stderr.lines) == []
    stdout = result.stdout.str()
    assert 'test_success_before_fail' not in stdout
    assert 'test_fail_on_flag PASSED' in stdout
    assert 'test_success_after_fail PASSED' in stdout

@pytest.mark.parametrize('stepwise_skip', ['--stepwise-skip', '--sw-skip'])
def test_run_with_skip_option(stepwise_pytester: Pytester, stepwise_skip: str) -> None:
    if False:
        while True:
            i = 10
    result = stepwise_pytester.runpytest('-v', '--strict-markers', '--stepwise', stepwise_skip, '--fail', '--fail-last')
    assert _strip_resource_warnings(result.stderr.lines) == []
    stdout = result.stdout.str()
    assert 'test_fail_on_flag FAILED' in stdout
    assert 'test_success_after_fail PASSED' in stdout
    assert 'test_fail_last_on_flag FAILED' in stdout
    assert 'test_success_after_last_fail' not in stdout

def test_fail_on_errors(error_pytester: Pytester) -> None:
    if False:
        return 10
    result = error_pytester.runpytest('-v', '--strict-markers', '--stepwise')
    assert _strip_resource_warnings(result.stderr.lines) == []
    stdout = result.stdout.str()
    assert 'test_error ERROR' in stdout
    assert 'test_success_after_fail' not in stdout

def test_change_testfile(stepwise_pytester: Pytester) -> None:
    if False:
        return 10
    result = stepwise_pytester.runpytest('-v', '--strict-markers', '--stepwise', '--fail', 'test_a.py')
    assert _strip_resource_warnings(result.stderr.lines) == []
    stdout = result.stdout.str()
    assert 'test_fail_on_flag FAILED' in stdout
    result = stepwise_pytester.runpytest('-v', '--strict-markers', '--stepwise', 'test_b.py')
    assert _strip_resource_warnings(result.stderr.lines) == []
    stdout = result.stdout.str()
    assert 'test_success PASSED' in stdout

@pytest.mark.parametrize('broken_first', [True, False])
def test_stop_on_collection_errors(broken_pytester: Pytester, broken_first: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Stop during collection errors. Broken test first or broken test last\n    actually surfaced a bug (#5444), so we test both situations.'
    files = ['working_testfile.py', 'broken_testfile.py']
    if broken_first:
        files.reverse()
    result = broken_pytester.runpytest('-v', '--strict-markers', '--stepwise', *files)
    result.stdout.fnmatch_lines('*error during collection*')

def test_xfail_handling(pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Ensure normal xfail is ignored, and strict xfail interrupts the session in sw mode\n\n    (#5547)\n    '
    monkeypatch.setattr('sys.dont_write_bytecode', True)
    contents = '\n        import pytest\n        def test_a(): pass\n\n        @pytest.mark.xfail(strict={strict})\n        def test_b(): assert {assert_value}\n\n        def test_c(): pass\n        def test_d(): pass\n    '
    pytester.makepyfile(contents.format(assert_value='0', strict='False'))
    result = pytester.runpytest('--sw', '-v')
    result.stdout.fnmatch_lines(['*::test_a PASSED *', '*::test_b XFAIL *', '*::test_c PASSED *', '*::test_d PASSED *', '* 3 passed, 1 xfailed in *'])
    pytester.makepyfile(contents.format(assert_value='1', strict='True'))
    result = pytester.runpytest('--sw', '-v')
    result.stdout.fnmatch_lines(['*::test_a PASSED *', '*::test_b FAILED *', '* Interrupted*', '* 1 failed, 1 passed in *'])
    pytester.makepyfile(contents.format(assert_value='0', strict='True'))
    result = pytester.runpytest('--sw', '-v')
    result.stdout.fnmatch_lines(['*::test_b XFAIL *', '*::test_c PASSED *', '*::test_d PASSED *', '* 2 passed, 1 deselected, 1 xfailed in *'])

def test_stepwise_skip_is_independent(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        def test_one():\n            assert False\n\n        def test_two():\n            assert False\n\n        def test_three():\n            assert False\n\n        ')
    result = pytester.runpytest('--tb', 'no', '--stepwise-skip')
    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(['FAILED test_stepwise_skip_is_independent.py::test_one - assert False', 'FAILED test_stepwise_skip_is_independent.py::test_two - assert False', '*Interrupted: Test failed, continuing from this test next run.*'])

def test_sw_skip_help(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    result = pytester.runpytest('-h')
    result.stdout.fnmatch_lines('*Implicitly enables --stepwise.')

def test_stepwise_xdist_dont_store_lastfailed(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makefile(ext='.ini', pytest=f'[pytest]\ncache_dir = {pytester.path}\n')
    pytester.makepyfile(conftest='\nimport pytest\n\n@pytest.hookimpl(tryfirst=True)\ndef pytest_configure(config) -> None:\n    config.workerinput = True\n')
    pytester.makepyfile(test_one='\ndef test_one():\n    assert False\n')
    result = pytester.runpytest('--stepwise')
    assert result.ret == pytest.ExitCode.INTERRUPTED
    stepwise_cache_file = pytester.path / Cache._CACHE_PREFIX_VALUES / STEPWISE_CACHE_DIR
    assert not Path(stepwise_cache_file).exists()

def test_disabled_stepwise_xdist_dont_clear_cache(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makefile(ext='.ini', pytest=f'[pytest]\ncache_dir = {pytester.path}\n')
    stepwise_cache_file = pytester.path / Cache._CACHE_PREFIX_VALUES / STEPWISE_CACHE_DIR
    stepwise_cache_dir = stepwise_cache_file.parent
    stepwise_cache_dir.mkdir(exist_ok=True, parents=True)
    stepwise_cache_file_relative = f'{Cache._CACHE_PREFIX_VALUES}/{STEPWISE_CACHE_DIR}'
    expected_value = '"test_one.py::test_one"'
    content = {f'{stepwise_cache_file_relative}': expected_value}
    pytester.makefile(ext='', **content)
    pytester.makepyfile(conftest='\nimport pytest\n\n@pytest.hookimpl(tryfirst=True)\ndef pytest_configure(config) -> None:\n    config.workerinput = True\n')
    pytester.makepyfile(test_one='\ndef test_one():\n    assert True\n')
    result = pytester.runpytest()
    assert result.ret == 0
    assert Path(stepwise_cache_file).exists()
    with stepwise_cache_file.open(encoding='utf-8') as file_handle:
        observed_value = file_handle.readlines()
    assert [expected_value] == observed_value