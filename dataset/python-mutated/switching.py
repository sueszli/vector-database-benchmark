from __future__ import absolute_import, print_function, division
from functools import wraps
from gevent.hub import _get_hub
from .hub import QuietHub
from .patched_tests_setup import get_switch_expected

def wrap_switch_count_check(method):
    if False:
        return 10

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        initial_switch_count = getattr(_get_hub(), 'switch_count', None)
        self.switch_expected = getattr(self, 'switch_expected', True)
        if initial_switch_count is not None:
            fullname = getattr(self, 'fullname', None)
            if self.switch_expected == 'default' and fullname:
                self.switch_expected = get_switch_expected(fullname)
        result = method(self, *args, **kwargs)
        if initial_switch_count is not None and self.switch_expected is not None:
            switch_count = _get_hub().switch_count - initial_switch_count
            if self.switch_expected is True:
                assert switch_count >= 0
                if not switch_count:
                    raise AssertionError('%s did not switch' % fullname)
            elif self.switch_expected is False:
                if switch_count:
                    raise AssertionError('%s switched but not expected to' % fullname)
            else:
                raise AssertionError('Invalid value for switch_expected: %r' % (self.switch_expected,))
        return result
    return wrapper

class CountingHub(QuietHub):
    switch_count = 0

    def switch(self, *args):
        if False:
            return 10
        self.switch_count += 1
        return QuietHub.switch(self, *args)