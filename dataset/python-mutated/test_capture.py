import pytest
from hypothesis.internal.compat import WINDOWS, escape_unicode_characters
pytest_plugins = 'pytester'
TESTSUITE = '\nfrom hypothesis import given, settings, Verbosity\nfrom hypothesis.strategies import integers\n\n@settings(verbosity=Verbosity.verbose)\n@given(integers())\ndef test_should_be_verbose(x):\n    pass\n\n'

@pytest.mark.parametrize('capture,expected', [('no', True), ('fd', False)])
def test_output_without_capture(testdir, capture, expected):
    if False:
        for i in range(10):
            print('nop')
    script = testdir.makepyfile(TESTSUITE)
    result = testdir.runpytest(script, '--verbose', '--capture', capture)
    out = '\n'.join(result.stdout.lines)
    assert 'test_should_be_verbose' in out
    assert ('Trying example' in out) == expected
    assert result.ret == 0
UNICODE_EMITTING = '\nimport pytest\nfrom hypothesis import given, settings, Verbosity\nfrom hypothesis.strategies import text\nimport sys\n\ndef test_emits_unicode():\n    @settings(verbosity=Verbosity.verbose)\n    @given(text())\n    def test_should_emit_unicode(t):\n        assert all(ord(c) <= 1000 for c in t), ascii(t)\n    with pytest.raises(AssertionError):\n        test_should_emit_unicode()\n'

@pytest.mark.xfail(WINDOWS, reason="Encoding issues in running the subprocess, possibly pytest's fault", strict=False)
def test_output_emitting_unicode(testdir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('LC_ALL', 'C')
    monkeypatch.setenv('LANG', 'C')
    script = testdir.makepyfile(UNICODE_EMITTING)
    result = getattr(testdir, 'runpytest_subprocess', testdir.runpytest)(script, '--verbose', '--capture=no')
    out = '\n'.join(result.stdout.lines)
    assert 'test_emits_unicode' in out
    assert chr(1001) in out or escape_unicode_characters(chr(1001)) in out
    assert result.ret == 0

def get_line_num(token, result, skip_n=0):
    if False:
        while True:
            i = 10
    skipped = 0
    for (i, line) in enumerate(result.stdout.lines):
        if token in line:
            if skip_n == skipped:
                return i
            else:
                skipped += 1
    raise AssertionError(f'Token {token!r} not found (skipped {skipped} of planned {skip_n} skips)')
TRACEBACKHIDE_HEALTHCHECK = '\nfrom hypothesis import given, settings\nfrom hypothesis.strategies import integers\nimport time\n@given(integers().map(lambda x: time.sleep(0.2)))\ndef test_healthcheck_traceback_is_hidden(x):\n    pass\n'

def test_healthcheck_traceback_is_hidden(testdir):
    if False:
        i = 10
        return i + 15
    script = testdir.makepyfile(TRACEBACKHIDE_HEALTHCHECK)
    result = testdir.runpytest(script, '--verbose')
    def_token = '__ test_healthcheck_traceback_is_hidden __'
    timeout_token = ': FailedHealthCheck'
    def_line = get_line_num(def_token, result)
    timeout_line = get_line_num(timeout_token, result)
    assert timeout_line - def_line == 7
COMPOSITE_IS_NOT_A_TEST = '\nfrom hypothesis.strategies import composite, none\n@composite\ndef test_data_factory(draw):\n    return draw(none())\n'

def test_deprecation_of_strategies_as_tests(testdir):
    if False:
        print('Hello World!')
    script = testdir.makepyfile(COMPOSITE_IS_NOT_A_TEST)
    testdir.runpytest(script).assert_outcomes(failed=1)