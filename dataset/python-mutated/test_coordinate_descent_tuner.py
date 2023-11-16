import sys
import unittest
from unittest import mock
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA
try:
    import triton
except ImportError:
    if __name__ == '__main__':
        sys.exit(0)
    raise unittest.SkipTest('requires triton')
from torch._inductor import config
from torch._inductor.coordinate_descent_tuner import CoordescTuner
config.benchmark_kernel = True
config.coordinate_descent_tuning = True
orig_compare_config = CoordescTuner.compare_config

def mock_compare_config_prefer_larger_XBLOCK(self, func, candidate_config, best_config, best_timing):
    if False:
        for i in range(10):
            print('nop')
    '\n    self is the CoordescTuner object\n    '
    if 'XBLOCK' in candidate_config.kwargs:
        assert 'XBLOCK' in best_config.kwargs
        if candidate_config.kwargs['XBLOCK'] < best_config.kwargs['XBLOCK']:
            func(candidate_config)
            return (False, best_timing * 1.1)
        elif candidate_config.kwargs['XBLOCK'] > best_config.kwargs['XBLOCK']:
            func(candidate_config)
            return (True, best_timing * 0.9)
    return orig_compare_config(self, func, candidate_config, best_config, best_timing)

class TestCoordinateDescentTuner(TestCase):

    def test_abs_function(self):
        if False:
            print('Hello World!')
        '\n        The benchmark result is simply abs(XBLOCK - 15)\n        '
        tuner = CoordescTuner()
        baseline_config = triton.Config({'XBLOCK': 1}, num_warps=8, num_stages=1)

        def func(config):
            if False:
                return 10
            return abs(config.kwargs['XBLOCK'] - 15)
        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get('XBLOCK') == 16, str(best_config))

    def test_no_neighbors(self):
        if False:
            print('Hello World!')
        '\n        Test the case that there is no available neighbor values for a field.\n        '
        tuner = CoordescTuner(size_hints=[1])
        baseline_config = triton.Config({'XBLOCK': 1}, num_warps=8, num_stages=1)

        def func(config):
            if False:
                for i in range(10):
                    print('nop')
            return abs(config.kwargs['XBLOCK'] - 15)
        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get('XBLOCK') == 1, str(best_config))

    def test_get_neighbour_values(self):
        if False:
            while True:
                i = 10
        tuner = CoordescTuner()
        neighbours = tuner.get_neighbour_values('num_stages', 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})
        neighbours = tuner.get_neighbour_values('num_warps', 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})

    def test_persistent_reduction(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            return x / x.sum(dim=-1, keepdim=True)
        with mock.patch.object(CoordescTuner, 'compare_config', mock_compare_config_prefer_larger_XBLOCK):
            x = torch.ones(2, 256).cuda()
            expected = f(x)
            _ = torch.compile(f)(x)
            actual = torch.compile(f)(x)
            self.assertTrue(torch.allclose(expected, actual, atol=0.0001, rtol=0.0001), f'Expected:\n{expected}\nActual:\n{actual}')
if __name__ == '__main__':
    if IS_LINUX and HAS_CUDA:
        run_tests()