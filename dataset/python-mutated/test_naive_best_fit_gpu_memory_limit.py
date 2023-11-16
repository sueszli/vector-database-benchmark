import unittest
import numpy as np
from paddle import base
base.core.globals()['FLAGS_allocator_strategy'] = 'naive_best_fit'
if base.is_compiled_with_cuda():
    base.core.globals()['FLAGS_gpu_memory_limit_mb'] = 10

class TestBase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if base.is_compiled_with_cuda():
            self._limit = base.core.globals()['FLAGS_gpu_memory_limit_mb']

    def test_allocate(self):
        if False:
            i = 10
            return i + 15
        if not base.is_compiled_with_cuda():
            return
        other_dim = int(1024 * 1024 / 4)
        place = base.CUDAPlace(0)
        t = base.LoDTensor()
        t.set(np.ndarray([int(self._limit / 2), other_dim], dtype='float32'), place)
        del t
        t = base.LoDTensor()
        large_np = np.ndarray([2 * self._limit, other_dim], dtype='float32')
        try:
            t.set(large_np, place)
            self.assertTrue(False)
        except:
            self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()