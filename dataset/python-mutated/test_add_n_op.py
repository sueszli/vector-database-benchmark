import unittest
import numpy as np
import paddle

class TestAddnOp(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(20)
        l = 32
        self.x_np = np.random.random([l, 16, 256])

    def check_main(self, x_np, dtype, axis=None):
        if False:
            return 10
        paddle.disable_static()
        x = []
        for i in range(x_np.shape[0]):
            val = paddle.to_tensor(x_np[i].astype(dtype))
            val.stop_gradient = False
            x.append(val)
        y = paddle.add_n(x)
        x_g = paddle.grad(y, x)
        y_np = y.numpy().astype('float32')
        x_g_np = []
        for val in x_g:
            x_g_np.append(val.numpy().astype('float32'))
        paddle.enable_static()
        return (y_np, x_g_np)

    def test_add_n_fp16(self):
        if False:
            print('Hello World!')
        if not paddle.is_compiled_with_cuda():
            return
        (y_np_16, x_g_np_16) = self.check_main(self.x_np, 'float16')
        (y_np_32, x_g_np_32) = self.check_main(self.x_np, 'float32')
        np.testing.assert_allclose(y_np_16, y_np_32, rtol=0.001)
        for i in range(len(x_g_np_32)):
            np.testing.assert_allclose(x_g_np_16[i], x_g_np_32[i], rtol=0.001)

    def test_add_n_api(self):
        if False:
            for i in range(10):
                print('nop')
        if not paddle.is_compiled_with_cuda():
            return
        (y_np_32, x_g_np_32) = self.check_main(self.x_np, 'float32')
        y_np_gt = np.sum(self.x_np, axis=0).astype('float32')
        np.testing.assert_allclose(y_np_32, y_np_gt, rtol=1e-06)
if __name__ == '__main__':
    unittest.main()