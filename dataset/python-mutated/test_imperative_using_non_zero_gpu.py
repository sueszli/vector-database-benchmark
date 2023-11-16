import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base.dygraph import guard, to_variable

class TestImperativeUsingNonZeroGpu(unittest.TestCase):

    def run_main(self, np_arr, place):
        if False:
            while True:
                i = 10
        with guard(place):
            var = to_variable(np_arr)
            np.testing.assert_array_equal(np_arr, var.numpy())

    def test_non_zero_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        if not base.is_compiled_with_cuda():
            return
        np_arr = np.random.random([11, 13]).astype('float32')
        if paddle.device.cuda.device_count() > 1:
            self.run_main(np_arr, base.CUDAPlace(1))
        else:
            self.run_main(np_arr, base.CUDAPlace(0))
if __name__ == '__main__':
    unittest.main()