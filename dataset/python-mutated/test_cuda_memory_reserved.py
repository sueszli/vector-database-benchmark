import unittest
import paddle
from paddle.base import core
from paddle.device.cuda import device_count, memory_reserved

class TestMemoryreserved(unittest.TestCase):

    def func_test_memory_reserved(self, device=None):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            alloc_size = 4 * 256
            memory_reserved_size = memory_reserved(device)
            self.assertEqual(memory_reserved_size, alloc_size)

    def test_memory_reserved_for_all_places(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device('gpu:' + str(i))
                self.func_test_memory_reserved(core.CUDAPlace(i))
                self.func_test_memory_reserved(i)
                self.func_test_memory_reserved('gpu:' + str(i))

    def test_memory_reserved_exception(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            wrong_device = [core.CPUPlace(), device_count() + 1, -2, 0.5, 'gpu1']
            for device in wrong_device:
                with self.assertRaises(BaseException):
                    memory_reserved(device)
        else:
            with self.assertRaises(ValueError):
                memory_reserved()
if __name__ == '__main__':
    unittest.main()