import unittest
import paddle
from paddle.base import core
from paddle.device.cuda import device_count, memory_allocated

class TestMemoryAllocated(unittest.TestCase):

    def test_memory_allocated(self, device=None):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            alloc_size = 4 * 256
            memory_allocated_size = memory_allocated(device)
            self.assertEqual(memory_allocated_size, alloc_size)

    def test_memory_allocated_for_all_places(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device('gpu:' + str(i))
                self.test_memory_allocated(core.CUDAPlace(i))
                self.test_memory_allocated(i)
                self.test_memory_allocated('gpu:' + str(i))

    def test_memory_allocated_exception(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            wrong_device = [core.CPUPlace(), device_count() + 1, -2, 0.5, 'gpu1']
            for device in wrong_device:
                with self.assertRaises(BaseException):
                    memory_allocated(device)
        else:
            with self.assertRaises(ValueError):
                memory_allocated()
if __name__ == '__main__':
    unittest.main()