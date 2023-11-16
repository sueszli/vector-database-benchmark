import sys
import pytest
from hypothesis.internal.compat import PYPY
from hypothesis.internal.scrutineer import make_report
pytestmark = pytest.mark.skipif(PYPY or sys.gettrace(), reason='See comment')
BUG_MARKER = '# BUG'
DEADLINE_PRELUDE = '\nfrom datetime import timedelta\nfrom hypothesis.errors import DeadlineExceeded\n'
PRELUDE = '\nfrom hypothesis import Phase, given, settings, strategies as st\n\n@settings(phases=tuple(Phase), derandomize=True)\n'
TRIVIAL = '\n@given(st.integers())\ndef test_reports_branch_in_test(x):\n    if x > 10:\n        raise AssertionError  # BUG\n'
MULTIPLE_BUGS = '\n@given(st.integers(), st.integers())\ndef test_reports_branch_in_test(x, y):\n    if x > 10:\n        raise (AssertionError if x % 2 else Exception)  # BUG\n'
FRAGMENTS = (pytest.param(TRIVIAL, id='trivial'), pytest.param(MULTIPLE_BUGS, id='multiple-bugs'))

def get_reports(file_contents, *, testdir):
    if False:
        for i in range(10):
            print('nop')
    test_file = str(testdir.makepyfile(file_contents))
    pytest_stdout = str(testdir.runpytest_inprocess(test_file, '--tb=native').stdout)
    explanations = {i: {(test_file, i)} for (i, line) in enumerate(file_contents.splitlines()) if line.endswith(BUG_MARKER)}
    expected = [('\n'.join(r), '\n    | '.join(r)) for r in make_report(explanations).values()]
    return (pytest_stdout, expected)

@pytest.mark.parametrize('code', FRAGMENTS)
def test_explanations(code, testdir):
    if False:
        print('Hello World!')
    (pytest_stdout, expected) = get_reports(PRELUDE + code, testdir=testdir)
    assert len(expected) == code.count(BUG_MARKER)
    for (single, group) in expected:
        assert single in pytest_stdout or group in pytest_stdout

@pytest.mark.parametrize('code', FRAGMENTS)
def test_no_explanations_if_deadline_exceeded(code, testdir):
    if False:
        return 10
    code = code.replace('AssertionError', 'DeadlineExceeded(timedelta(), timedelta())')
    (pytest_stdout, _) = get_reports(DEADLINE_PRELUDE + PRELUDE + code, testdir=testdir)
    assert 'Explanation:' not in pytest_stdout
NO_SHOW_CONTEXTLIB = '\nfrom contextlib import contextmanager\nfrom hypothesis import given, strategies as st, Phase, settings\n\n@contextmanager\ndef ctx():\n    yield\n\n@settings(phases=list(Phase))\n@given(st.integers())\ndef test(x):\n    with ctx():\n        assert x < 100\n'

@pytest.mark.skipif(PYPY, reason='Tracing is slow under PyPy')
def test_skips_uninformative_locations(testdir):
    if False:
        print('Hello World!')
    (pytest_stdout, _) = get_reports(NO_SHOW_CONTEXTLIB, testdir=testdir)
    assert 'Explanation:' not in pytest_stdout