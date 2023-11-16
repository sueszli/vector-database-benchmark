import unittest
import pytest
from hypothesis import given
from hypothesis.core import skip_exceptions_to_reraise
from hypothesis.strategies import integers
from tests.common.utils import capture_out

@pytest.mark.parametrize('skip_exception', skip_exceptions_to_reraise())
def test_no_falsifying_example_if_unittest_skip(skip_exception):
    if False:
        for i in range(10):
            print('nop')
    'If a ``SkipTest`` exception is raised during a test, Hypothesis should\n    not continue running the test and shrink process, nor should it print\n    anything about falsifying examples.'

    class DemoTest(unittest.TestCase):

        @given(xs=integers())
        def test_to_be_skipped(self, xs):
            if False:
                print('Hello World!')
            if xs == 0:
                raise skip_exception
            else:
                assert xs == 0
    with capture_out() as o:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DemoTest)
        unittest.TextTestRunner().run(suite)
    assert 'Falsifying example' not in o.getvalue()