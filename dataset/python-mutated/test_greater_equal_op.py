import unittest
import numpy as np
import paddle
from paddle import static
from paddle.base import core

class Test_Greater_Equal_Op_Fp16(unittest.TestCase):

    def test_api_fp16(self):
        if False:
            return 10
        paddle.enable_static()
        with static.program_guard(static.Program(), static.Program()):
            label = paddle.to_tensor([3, 3], dtype='float16')
            limit = paddle.to_tensor([3, 2], dtype='float16')
            out = paddle.greater_equal(x=label, y=limit)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = static.Executor(place)
                (res,) = exe.run(fetch_list=[out])
                self.assertEqual((res == np.array([True, True])).all(), True)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()