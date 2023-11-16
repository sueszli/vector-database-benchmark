import unittest
import numpy as np
import paddle
from paddle import base
if base.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth', 'FLAGS_auto_growth_chunk_size_in_mb': 10})

class TestMemoryLimit(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._limit = 10
        if base.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_gpu_memory_limit_mb': 10})

    def test_allocate(self):
        if False:
            return 10
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

class TestChunkSize(unittest.TestCase):

    def test_allocate(self):
        if False:
            while True:
                i = 10
        if not base.is_compiled_with_cuda():
            return
        paddle.rand([1024])
        (reserved, allocated) = (paddle.device.cuda.max_memory_reserved(), paddle.device.cuda.max_memory_allocated())
        self.assertEqual(reserved, 1024 * 1024 * 10)
        self.assertEqual(allocated, 1024 * 4)
if __name__ == '__main__':
    unittest.main()