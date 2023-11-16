import sys
import pathlib
import pytest
from scripts.dev import check_coverage
pytest_plugins = 'pytester'
pytestmark = [pytest.mark.linux, pytest.mark.not_frozen]

class CovtestHelper:
    """Helper object for covtest fixture.

    Attributes:
        _testdir: The testdir fixture from pytest.
        _monkeypatch: The monkeypatch fixture from pytest.
    """

    def __init__(self, testdir, monkeypatch):
        if False:
            i = 10
            return i + 15
        self._testdir = testdir
        self._monkeypatch = monkeypatch

    def makefile(self, code):
        if False:
            while True:
                i = 10
        'Generate a module.py for the given code.'
        self._testdir.makepyfile(module=code)

    def run(self):
        if False:
            print('Hello World!')
        'Run pytest with coverage for the given module.py.'
        coveragerc = str(self._testdir.tmpdir / 'coveragerc')
        self._monkeypatch.delenv('PYTEST_ADDOPTS', raising=False)
        res = self._testdir.runpytest('--cov=module', '--cov-config={}'.format(coveragerc), '--cov-report=xml', plugins=['no:faulthandler', 'no:xvfb'])
        assert res.ret == 0
        return res

    def check(self, perfect_files=None):
        if False:
            i = 10
            return i + 15
        'Run check_coverage.py and run its return value.'
        coverage_file = self._testdir.tmpdir / 'coverage.xml'
        if perfect_files is None:
            perfect_files = [(None, 'module.py')]
        argv = [sys.argv[0]]
        self._monkeypatch.setattr(check_coverage.sys, 'argv', argv)
        with self._testdir.tmpdir.as_cwd():
            with coverage_file.open(encoding='utf-8') as f:
                return check_coverage.check(f, perfect_files=perfect_files)

    def check_skipped(self, args, reason):
        if False:
            return 10
        "Run check_coverage.py and make sure it's skipped."
        argv = [sys.argv[0]] + list(args)
        self._monkeypatch.setattr(check_coverage.sys, 'argv', argv)
        with pytest.raises(check_coverage.Skipped) as excinfo:
            check_coverage.check(None, perfect_files=[])
        assert excinfo.value.reason == reason

@pytest.fixture
def covtest(testdir, monkeypatch):
    if False:
        while True:
            i = 10
    'Fixture which provides a coveragerc and a test to call module.func.'
    testdir.makefile(ext='', coveragerc='\n        [run]\n        branch=True\n    ')
    testdir.makepyfile(test_module='\n        from module import func\n\n        def test_module():\n            func()\n    ')
    res = testdir.runpytest('--version', '--version')
    assert res.ret == 0
    output = res.stderr.str()
    if not output:
        output = res.stdout.str()
    assert 'This is pytest version' in output
    if 'pytest-cov' not in output:
        pytest.skip('cov plugin not available')
    return CovtestHelper(testdir, monkeypatch)

def test_tested_no_branches(covtest):
    if False:
        while True:
            i = 10
    covtest.makefile('\n        def func():\n            pass\n    ')
    covtest.run()
    assert covtest.check() == []

def test_tested_with_branches(covtest):
    if False:
        for i in range(10):
            print('nop')
    covtest.makefile('\n        def func2(arg):\n            if arg:\n                pass\n            else:\n                pass\n\n        def func():\n            func2(True)\n            func2(False)\n    ')
    covtest.run()
    assert covtest.check() == []

def test_untested(covtest):
    if False:
        print('Hello World!')
    covtest.makefile('\n        def func():\n            pass\n\n        def untested():\n            pass\n    ')
    covtest.run()
    expected = check_coverage.Message(check_coverage.MsgType.insufficient_coverage, 'module.py', 'module.py has 75.00% line and 100.00% branch coverage!')
    assert covtest.check() == [expected]

def test_untested_floats(covtest):
    if False:
        for i in range(10):
            print('nop')
    "Make sure we don't report 58.330000000000005% coverage."
    covtest.makefile('\n        def func():\n            pass\n\n        def untested():\n            pass\n\n        def untested2():\n            pass\n\n        def untested3():\n            pass\n\n        def untested4():\n            pass\n\n        def untested5():\n            pass\n    ')
    covtest.run()
    expected = check_coverage.Message(check_coverage.MsgType.insufficient_coverage, 'module.py', 'module.py has 58.33% line and 100.00% branch coverage!')
    assert covtest.check() == [expected]

def test_untested_branches(covtest):
    if False:
        for i in range(10):
            print('nop')
    covtest.makefile('\n        def func2(arg):\n            if arg:\n                pass\n            else:\n                pass\n\n        def func():\n            func2(True)\n    ')
    covtest.run()
    line_coverage = '83.33%' if sys.version_info[:2] >= (3, 10) else '100.00%'
    expected = check_coverage.Message(check_coverage.MsgType.insufficient_coverage, 'module.py', f'module.py has {line_coverage} line and 50.00% branch coverage!')
    assert covtest.check() == [expected]

def test_tested_unlisted(covtest):
    if False:
        while True:
            i = 10
    covtest.makefile('\n        def func():\n            pass\n    ')
    covtest.run()
    expected = check_coverage.Message(check_coverage.MsgType.perfect_file, 'module.py', 'module.py has 100% coverage but is not in perfect_files!')
    assert covtest.check(perfect_files=[]) == [expected]

@pytest.mark.parametrize('args, reason', [(['-k', 'foo'], 'because -k is given.'), (['-m', 'foo'], 'because -m is given.'), (['--lf'], 'because --lf is given.'), (['blah', '-m', 'foo'], 'because -m is given.'), (['tests/foo'], 'because there is nothing to check.')])
def test_skipped_args(covtest, args, reason):
    if False:
        for i in range(10):
            print('nop')
    covtest.check_skipped(args, reason)

@pytest.mark.fake_os('windows')
def test_skipped_non_linux(covtest):
    if False:
        while True:
            i = 10
    covtest.check_skipped([], 'on non-Linux system.')

def _generate_files():
    if False:
        while True:
            i = 10
    'Get filenames from WHITELISTED_/PERFECT_FILES.'
    for src_file in check_coverage.WHITELISTED_FILES:
        yield pathlib.Path(src_file)
    for (test_file, src_file) in check_coverage.PERFECT_FILES:
        if test_file is not None:
            yield pathlib.Path(test_file)
        yield pathlib.Path(src_file)

@pytest.mark.parametrize('filename', list(_generate_files()))
def test_files_exist(filename):
    if False:
        return 10
    basedir = pathlib.Path(check_coverage.__file__).parents[2]
    assert (basedir / filename).exists()