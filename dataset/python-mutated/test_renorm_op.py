import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import Program, program_guard
paddle.set_device('cpu')

class TestRenormAPI(unittest.TestCase):

    def input_data(self):
        if False:
            i = 10
            return i + 15
        self.data_x = np.array([[[2.0, 2, -2], [3, 0.3, 3]], [[2, -8, 2], [3.1, 3.7, 3]]])
        self.p = 1.0
        self.dim = 2
        self.max_norm = 2.05

    def test_renorm_api(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        self.input_data()
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 2, 3], dtype='float64')
            z = paddle.renorm(x, self.p, self.dim, self.max_norm)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(feed={'x': self.data_x}, fetch_list=[z], return_numpy=False)
        expected = np.array([[[0.40594056, 0.29285714, -0.41], [0.60891086, 0.04392857, 0.61500001]], [[0.40594056, -1.17142856, 0.41], [0.62920785, 0.54178572, 0.61500001]]])
        np.testing.assert_allclose(expected, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        if False:
            i = 10
            return i + 15
        self.input_data()
        with base.dygraph.guard(base.CPUPlace()):
            input = [[[2.0, 2, -2], [3, 0.3, 3]], [[2, -8, 2], [3.1, 3.7, 3]]]
            x = paddle.to_tensor(input, stop_gradient=False)
            y = paddle.renorm(x, 1.0, 2, 2.05)
            expected = np.array([[[0.40594056, 0.29285714, -0.41], [0.60891086, 0.04392857, 0.61500001]], [[0.40594056, -1.17142856, 0.41], [0.62920785, 0.54178572, 0.61500001]]])
            np.testing.assert_allclose(expected, np.array(y), rtol=1e-05)
            z = paddle.mean(y)
            z.backward(retain_graph=True)
            expected_grad = np.array([[[0, 0.01394558, 0.02733333], [0, 0.01394558, 0.00683333]], [[0, 0.01045918, 0.00683333], [0, 0.01394558, 0.00683333]]])
            np.testing.assert_allclose(expected_grad, np.array(x.grad), rtol=1e-05)
        with base.dygraph.guard():
            input = [[[2.0, 2, -2], [3, 0.3, 3]], [[2, -8, 2], [3.1, 3.7, 3]]]
            x = paddle.to_tensor(input, stop_gradient=False)
            exp = False
            try:
                paddle.renorm(x, 1.0, 8, 2.05)
            except:
                exp = True
            self.assertTrue(exp)
            exp = False
            try:
                paddle.renorm(x, 1.0, -4, 2.05)
            except:
                exp = True
            self.assertTrue(exp)
            y = paddle.renorm(x, 1.0, -1, 2.05)
            expected = np.array([[[0.40594056, 0.29285714, -0.41], [0.60891086, 0.04392857, 0.61500001]], [[0.40594056, -1.17142856, 0.41], [0.62920785, 0.54178572, 0.61500001]]])
            np.testing.assert_allclose(expected, np.array(y), rtol=1e-05)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()