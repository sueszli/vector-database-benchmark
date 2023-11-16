"""This test demonstrates the use of the "rate_limited" decorator.
You can use this decorator on any method to rate-limit it."""
from seleniumbase import BaseCase
from seleniumbase import decorators

class RateLimitingTests(BaseCase):

    @decorators.rate_limited(4.2)
    def print_item(self, item):
        if False:
            for i in range(10):
                print('nop')
        print(item)

    def test_rate_limited_printing(self):
        if False:
            i = 10
            return i + 15
        if self._multithreaded:
            self.open_if_not_url('about:blank')
            print('Skipping test in multi-threaded mode.')
            self.skip('Skipping test in multi-threaded mode.')
        if self.recorder_mode:
            self.open_if_not_url('about:blank')
            print('Skipping test in Recorder Mode.')
            self.skip('Skipping test in Recorder Mode.')
        message = 'Running rate-limited print() on the command line'
        self.open('data:text/html,<p>%s</p>' % message)
        print('\n%s:' % message)
        for item in range(1, 11):
            self.print_item(item)