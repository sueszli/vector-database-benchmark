import unittest
import paddle
from paddle.base import core
from paddle.device.cuda import device_count, max_memory_reserved, memory_reserved

class TestMaxMemoryreserved(unittest.TestCase):

    def test_max_memory_reserved(self, device=None):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            alloc_time = 100
            max_alloc_size = 10000
            peak_memory_reserved_size = max_memory_reserved(device)
            for i in range(alloc_time):
                shape = paddle.randint(max_alloc_size)
                tensor = paddle.zeros(shape)
                peak_memory_reserved_size = max(peak_memory_reserved_size, memory_reserved(device))
                del shape
                del tensor
            self.assertEqual(peak_memory_reserved_size, max_memory_reserved(device))

    def test_max_memory_reserved_for_all_places(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device('gpu:' + str(i))
                self.test_max_memory_reserved(core.CUDAPlace(i))
                self.test_max_memory_reserved(i)
                self.test_max_memory_reserved('gpu:' + str(i))

    def test_max_memory_reserved_exception(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            wrong_device = [core.CPUPlace(), device_count() + 1, -2, 0.5, 'gpu1']
            for device in wrong_device:
                with self.assertRaises(BaseException):
                    max_memory_reserved(device)
        else:
            with self.assertRaises(ValueError):
                max_memory_reserved()
if __name__ == '__main__':
    unittest.main()