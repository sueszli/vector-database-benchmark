import pytest
from _hypothesis_pytestplugin import PRINT_STATISTICS_OPTION
pytest_plugins = 'pytester'

def get_output(testdir, suite, *args):
    if False:
        return 10
    script = testdir.makepyfile(suite)
    result = testdir.runpytest(script, *args)
    return '\n'.join(result.stdout.lines)
TESTSUITE = "\nfrom hypothesis import HealthCheck, given, settings, assume\nfrom hypothesis.strategies import integers\nimport time\nimport warnings\nfrom hypothesis.errors import HypothesisDeprecationWarning\n\nwarnings.simplefilter('always', HypothesisDeprecationWarning)\n\n\n@given(integers())\ndef test_all_valid(x):\n    pass\n\n\n@settings(max_examples=100, suppress_health_check=list(HealthCheck))\n@given(integers())\ndef test_iterations(x):\n    assume(x == 13)\n"

def test_does_not_run_statistics_by_default(testdir):
    if False:
        while True:
            i = 10
    out = get_output(testdir, TESTSUITE)
    assert 'Hypothesis Statistics' not in out

def test_prints_statistics_given_option(testdir):
    if False:
        i = 10
        return i + 15
    out = get_output(testdir, TESTSUITE, PRINT_STATISTICS_OPTION)
    assert 'Hypothesis Statistics' in out
    assert 'max_examples=100' in out
    assert '< 10% of examples satisfied assumptions' in out

def test_prints_statistics_given_option_under_xdist(testdir):
    if False:
        for i in range(10):
            print('nop')
    out = get_output(testdir, TESTSUITE, PRINT_STATISTICS_OPTION, '-n', '2')
    assert 'Hypothesis Statistics' in out
    assert 'max_examples=100' in out
    assert '< 10% of examples satisfied assumptions' in out

def test_prints_statistics_given_option_with_junitxml(testdir):
    if False:
        return 10
    out = get_output(testdir, TESTSUITE, PRINT_STATISTICS_OPTION, '--junit-xml=out.xml')
    assert 'Hypothesis Statistics' in out
    assert 'max_examples=100' in out
    assert '< 10% of examples satisfied assumptions' in out

@pytest.mark.skipif(tuple(map(int, pytest.__version__.split('.')[:2])) < (5, 4), reason='too old')
def test_prints_statistics_given_option_under_xdist_with_junitxml(testdir):
    if False:
        while True:
            i = 10
    out = get_output(testdir, TESTSUITE, PRINT_STATISTICS_OPTION, '-n', '2', '--junit-xml=out.xml')
    assert 'Hypothesis Statistics' in out
    assert 'max_examples=100' in out
    assert '< 10% of examples satisfied assumptions' in out
UNITTEST_TESTSUITE = '\n\nfrom hypothesis import given\nfrom hypothesis.strategies import integers\nfrom unittest import TestCase\n\n\nclass TestStuff(TestCase):\n    @given(integers())\n    def test_all_valid(self, x):\n        pass\n'

def test_prints_statistics_for_unittest_tests(testdir):
    if False:
        return 10
    script = testdir.makepyfile(UNITTEST_TESTSUITE)
    result = testdir.runpytest(script, PRINT_STATISTICS_OPTION)
    out = '\n'.join(result.stdout.lines)
    assert 'Hypothesis Statistics' in out
    assert 'TestStuff::test_all_valid' in out
    assert 'max_examples=100' in out
STATEFUL_TESTSUITE = '\nfrom hypothesis.stateful import RuleBasedStateMachine, rule\n\nclass Stuff(RuleBasedStateMachine):\n    @rule()\n    def step(self):\n        pass\n\nTestStuff = Stuff.TestCase\n'

def test_prints_statistics_for_stateful_tests(testdir):
    if False:
        for i in range(10):
            print('nop')
    script = testdir.makepyfile(STATEFUL_TESTSUITE)
    result = testdir.runpytest(script, PRINT_STATISTICS_OPTION)
    out = '\n'.join(result.stdout.lines)
    assert 'Hypothesis Statistics' in out
    assert 'TestStuff::runTest' in out
    assert 'max_examples=100' in out