import pytest
from tests.support.case import SyndicCase

@pytest.mark.windows_whitelisted
@pytest.mark.skip(reason='The Syndic Tests are currently broken. See #58975')
class TestSyndic(SyndicCase):
    """
    Validate the syndic interface by testing the test module
    """

    @pytest.mark.slow_test
    def test_ping(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test.ping\n        '
        self.assertTrue(self.run_function('test.ping'))

    @pytest.mark.slow_test
    def test_fib(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test.fib\n        '
        self.assertEqual(self.run_function('test.fib', ['20'])[0], 6765)