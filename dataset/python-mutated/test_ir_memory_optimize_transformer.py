import os
import unittest
from paddle.base import core
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from test_parallel_executor_transformer import get_feed_data_reader, transformer

class TestTransformerWithIR(TestParallelExecutorBase):

    def test_main(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            self.check_network_convergence(transformer, use_device=DeviceType.CUDA, feed_data_reader=get_feed_data_reader(), use_ir_memory_optimize=False, iter=2)
            self.check_network_convergence(transformer, use_device=DeviceType.CUDA, feed_data_reader=get_feed_data_reader(), use_ir_memory_optimize=True, iter=2)
if __name__ == '__main__':
    unittest.main()