import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core

class TestTF32Switch(unittest.TestCase):

    def test_on_off(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self.assertTrue(core.get_cublas_switch())
            core.set_cublas_switch(False)
            self.assertFalse(core.get_cublas_switch())
            core.set_cublas_switch(True)
            self.assertTrue(core.get_cublas_switch())
            core.set_cublas_switch(True)
        else:
            pass

class TestTF32OnMatmul(unittest.TestCase):

    def test_dygraph_without_out(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            core.set_cublas_switch(False)
            with base.dygraph.guard(place):
                input_array1 = np.random.rand(4, 12, 64, 88).astype('float32')
                input_array2 = np.random.rand(4, 12, 88, 512).astype('float32')
                data1 = paddle.to_tensor(input_array1)
                data2 = paddle.to_tensor(input_array2)
                out = paddle.matmul(data1, data2)
                expected_result = np.matmul(input_array1, input_array2)
            np.testing.assert_allclose(expected_result, out.numpy(), rtol=0.001)
            core.set_cublas_switch(True)
        else:
            pass
if __name__ == '__main__':
    unittest.main()