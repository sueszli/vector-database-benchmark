import pytest
from seleniumbase import BaseCase
from seleniumbase import decorators
BaseCase.main(__name__, __file__)

class TimeLimitTests(BaseCase):

    @pytest.mark.expected_failure
    def test_runtime_limit_decorator(self):
        if False:
            i = 10
            return i + 15
        'This test fails on purpose to show the runtime_limit() decorator\n        for code blocks that run longer than the time limit specified.'
        print('\n(This test should fail)')
        self.open('https://xkcd.com/2511')
        with decorators.runtime_limit(0.7):
            self.sleep(0.95)

    @pytest.mark.expected_failure
    def test_set_time_limit_method(self):
        if False:
            print('Hello World!')
        "This test fails on purpose to show the set_time_limit() method\n        for tests that run longer than the time limit specified (seconds).\n        The time-limit clock starts after the browser has fully launched,\n        which is after pytest starts it's own internal clock for tests.\n        Usage: (inside tests) =>  self.set_time_limit(SECONDS)\n        Usage: (command-line) =>  --time-limit=SECONDS"
        self.set_time_limit(2.2)
        print('\n(This test should fail)')
        self.open('https://xkcd.com/1658')
        self.sleep(3)