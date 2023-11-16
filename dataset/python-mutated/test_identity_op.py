import unittest
import numpy as np
import paddle

class TestIdentityAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [4, 4]
        self.x = np.random.random((4, 4)).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            id_layer = paddle.nn.Identity()
            out = id_layer(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = self.x
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=1e-08)

    def test_api_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        id_layer = paddle.nn.Identity()
        out = id_layer(x_tensor)
        out_ref = self.x
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-08)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()