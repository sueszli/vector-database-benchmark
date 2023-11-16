pytest_plugins = 'pytester'
TESTSUITE = '\nfrom hypothesis import given\nfrom hypothesis.strategies import integers\n\n@given(integers())\ndef test_foo(x):\n    pass\n\ndef test_bar():\n    pass\n'

def test_can_select_mark(testdir):
    if False:
        for i in range(10):
            print('nop')
    script = testdir.makepyfile(TESTSUITE)
    result = testdir.runpytest(script, '--verbose', '--strict-markers', '-m', 'hypothesis')
    out = '\n'.join(result.stdout.lines)
    assert '1 passed, 1 deselected' in out
UNITTEST_TESTSUITE = '\nfrom hypothesis import given\nfrom hypothesis.strategies import integers\nfrom unittest import TestCase\n\nclass TestStuff(TestCase):\n    @given(integers())\n    def test_foo(self, x):\n        pass\n\n    def test_bar(self):\n        pass\n'

def test_can_select_mark_on_unittest(testdir):
    if False:
        return 10
    script = testdir.makepyfile(UNITTEST_TESTSUITE)
    result = testdir.runpytest(script, '--verbose', '--strict-markers', '-m', 'hypothesis')
    out = '\n'.join(result.stdout.lines)
    assert '1 passed, 1 deselected' in out