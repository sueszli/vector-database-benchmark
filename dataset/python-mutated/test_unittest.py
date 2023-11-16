import io
import sys
import unittest
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import FailedHealthCheck, HypothesisWarning
from tests.common.utils import fails_with

class Thing_with_a_subThing(unittest.TestCase):
    """Example test case using subTest for the actual test below."""

    @given(st.tuples(st.booleans(), st.booleans()))
    def thing(self, lst):
        if False:
            return 10
        for (i, b) in enumerate(lst):
            with pytest.warns(HypothesisWarning):
                with self.subTest((i, b)):
                    self.assertTrue(b)

def test_subTest():
    if False:
        return 10
    suite = unittest.TestSuite()
    suite.addTest(Thing_with_a_subThing('thing'))
    stream = io.StringIO()
    out = unittest.TextTestRunner(stream=stream).run(suite)
    assert len(out.failures) <= out.testsRun, out

class test_given_on_setUp_fails_health_check(unittest.TestCase):

    @fails_with(FailedHealthCheck)
    @given(st.integers())
    def setUp(self, i):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            while True:
                i = 10
        'Provide something to set up for, so the setUp method is called.'
SUBTEST_SUITE = '\nimport unittest\nfrom hypothesis import given, settings, strategies as st\n\nclass MyTest(unittest.TestCase):\n    @given(s=st.text())\n    @settings(deadline=None)\n    def test_subtest(self, s):\n        with self.subTest(text=s):\n            self.assertIsInstance(s, str)\n\nif __name__ == "__main__":\n    unittest.main()\n'

@pytest.mark.parametrize('err', [[], ['-Werror']])
def test_subTest_no_self(testdir, err):
    if False:
        print('Hello World!')
    fname = testdir.makepyfile(tests=SUBTEST_SUITE)
    result = testdir.run(sys.executable, *err, str(fname))
    expected = pytest.ExitCode.TESTS_FAILED if err else pytest.ExitCode.OK
    assert result.ret == expected, result.stderr.str()