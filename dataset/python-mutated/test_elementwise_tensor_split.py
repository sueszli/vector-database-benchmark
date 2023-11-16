import re
import unittest
import numpy as np
import paddle
from paddle.base import core

class TestElementwiseOp(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'elementwise_sub'
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = 'prim'

    def test_float16_sub(self):
        if False:
            return 10
        if not core.is_compiled_with_cuda():
            return
        gpu_info = paddle.device.cuda.get_device_properties()
        gpu_name = gpu_info.name
        try:
            re_result = re.split('[ , -]', gpu_name)
            memory = int(re_result[-1][:-2])
        except:
            memory = int(gpu_info.total_memory) // 1000 ** 3
        if memory < 37:
            return
        paddle.disable_static()
        tensor_a = paddle.rand(shape=[5120, 4, 384, 384], dtype='float16')
        tensor_b = paddle.rand(shape=[5120, 1, 384, 384], dtype='float16')
        tensor_z = paddle.subtract(tensor_a, tensor_b)
        (in0, in1) = paddle.split(tensor_a, num_or_sections=2, axis=1)
        (out0, out1) = paddle.split(tensor_z, num_or_sections=2, axis=1)
        split_add0 = paddle.subtract(tensor_b, in0)
        split_add1 = paddle.subtract(tensor_b, in1)
        result1 = paddle.any(paddle.equal(out0, split_add0), [0, 1, 2, 3])
        result2 = paddle.any(paddle.equal(out1, split_add1), [0, 1, 2, 3])
        np.testing.assert_equal(result1.numpy(), True)
        np.testing.assert_equal(result2.numpy(), True)
if __name__ == '__main__':
    unittest.main()