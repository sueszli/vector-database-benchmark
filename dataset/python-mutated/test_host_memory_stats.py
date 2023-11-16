import unittest
import paddle
from paddle.base import core
paddle.set_device('cpu')

class TestHostMemoryStats(unittest.TestCase):

    def test_memory_allocated_with_pinned(self, device=None):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            tensor_pinned = tensor.pin_memory()
            alloc_size = 4 * 256
            memory_allocated_size = core.host_memory_stat_current_value('Allocated', 0)
            self.assertEqual(memory_allocated_size, alloc_size * 2)

            def foo():
                if False:
                    print('Hello World!')
                tensor = paddle.zeros(shape=[256])
                tensor_pinned = tensor.pin_memory()
                memory_allocated_size = core.host_memory_stat_current_value('Allocated', 0)
                self.assertEqual(memory_allocated_size, alloc_size * 4)
                max_allocated_size = core.host_memory_stat_peak_value('Allocated', 0)
                self.assertEqual(memory_allocated_size, alloc_size * 4)
            foo()
            memory_allocated_size = core.host_memory_stat_current_value('Allocated', 0)
            self.assertEqual(memory_allocated_size, alloc_size * 2)
            max_allocated_size = core.host_memory_stat_peak_value('Allocated', 0)
            self.assertEqual(max_allocated_size, alloc_size * 4)
if __name__ == '__main__':
    unittest.main()