import unittest
import paddle
from paddle.base import core
from paddle.device.cuda import device_count, max_memory_allocated, memory_allocated

class TestMaxMemoryAllocated(unittest.TestCase):

    def func_test_max_memory_allocated(self, device=None):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            alloc_time = 100
            max_alloc_size = 10000
            peak_memory_allocated_size = max_memory_allocated(device)
            for i in range(alloc_time):
                shape = paddle.randint(max_alloc_size)
                tensor = paddle.zeros(shape)
                peak_memory_allocated_size = max(peak_memory_allocated_size, memory_allocated(device))
                del shape
                del tensor
            self.assertEqual(peak_memory_allocated_size, max_memory_allocated(device))

    def test_max_memory_allocated_for_all_places(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device('gpu:' + str(i))
                self.func_test_max_memory_allocated(core.CUDAPlace(i))
                self.func_test_max_memory_allocated(i)
                self.func_test_max_memory_allocated('gpu:' + str(i))

    def test_max_memory_allocated_exception(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            wrong_device = [core.CPUPlace(), device_count() + 1, -2, 0.5, 'gpu1']
            for device in wrong_device:
                with self.assertRaises(BaseException):
                    max_memory_allocated(device)
        else:
            with self.assertRaises(ValueError):
                max_memory_allocated()
if __name__ == '__main__':
    unittest.main()