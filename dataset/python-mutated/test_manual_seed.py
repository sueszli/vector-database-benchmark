import unittest
import numpy as np
import paddle
from paddle import base
from paddle.tensor import random

class TestManualSeed(unittest.TestCase):

    def test_seed(self):
        if False:
            return 10
        base.enable_dygraph()
        gen = paddle.seed(12312321111)
        x = random.gaussian([10], dtype='float32')
        st1 = gen.get_state()
        x1 = random.gaussian([10], dtype='float32')
        gen.set_state(st1)
        x2 = random.gaussian([10], dtype='float32')
        gen.manual_seed(12312321111)
        x3 = random.gaussian([10], dtype='float32')
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()
        if not base.core.is_compiled_with_cuda():
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()