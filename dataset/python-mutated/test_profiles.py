import pytest
from _hypothesis_pytestplugin import LOAD_PROFILE_OPTION
from hypothesis.version import __version__
pytest_plugins = 'pytester'
CONFTEST = '\nfrom hypothesis._settings import settings\nsettings.register_profile("test", settings(max_examples=1))\n'
TESTSUITE = '\nfrom hypothesis import given\nfrom hypothesis.strategies import integers\nfrom hypothesis._settings import settings\n\ndef test_this_one_is_ok():\n    assert settings().max_examples == 1\n'

def test_does_not_run_reporting_hook_by_default(testdir):
    if False:
        while True:
            i = 10
    script = testdir.makepyfile(TESTSUITE)
    testdir.makeconftest(CONFTEST)
    result = testdir.runpytest(script, LOAD_PROFILE_OPTION, 'test')
    out = '\n'.join(result.stdout.lines)
    assert '1 passed' in out
    assert 'hypothesis profile' not in out
    assert __version__ in out

@pytest.mark.parametrize('option', ['-v', '--hypothesis-verbosity=verbose'])
def test_runs_reporting_hook_in_any_verbose_mode(testdir, option):
    if False:
        for i in range(10):
            print('nop')
    script = testdir.makepyfile(TESTSUITE)
    testdir.makeconftest(CONFTEST)
    result = testdir.runpytest(script, LOAD_PROFILE_OPTION, 'test', option)
    out = '\n'.join(result.stdout.lines)
    assert '1 passed' in out
    assert 'max_examples=1' in out
    assert 'hypothesis profile' in out
    assert __version__ in out