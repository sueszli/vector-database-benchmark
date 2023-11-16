import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests
load_tests = load_tests

@unittest.skipIf(not torch.profiler.itt.is_available(), 'ITT is required')
class TestItt(TestCase):

    def test_itt(self):
        if False:
            print('Hello World!')
        torch.profiler.itt.range_push('foo')
        torch.profiler.itt.mark('bar')
        torch.profiler.itt.range_pop()
if __name__ == '__main__':
    run_tests()