import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.pir_utils import test_with_pir_api

class TestMVOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'mv'
        self.python_api = paddle.mv
        self.init_config()
        self.inputs = {'X': self.x, 'Vec': self.vec}
        self.outputs = {'Out': np.dot(self.x, self.vec)}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X', 'Vec'], 'Out', check_pir=True)

    def init_config(self):
        if False:
            return 10
        self.x = np.random.random((2, 100)).astype('float64')
        self.vec = np.random.random(100).astype('float64')

class TestMVAPI(unittest.TestCase):

    def test_dygraph_api_out(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.x_data = np.random.random((5, 100)).astype('float64')
        self.x = paddle.to_tensor(self.x_data)
        self.vec_data = np.random.random(100).astype('float64')
        self.vec = paddle.to_tensor(self.vec_data)
        z = paddle.mv(self.x, self.vec)
        np_z = z.numpy()
        z_expected = np.array(np.dot(self.x_data, self.vec_data))
        np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)
        paddle.enable_static()

    @test_with_pir_api
    def test_static_graph(self):
        if False:
            return 10
        for x_stop_gradient in [False, True]:
            for vec_stop_gradient in [False, True]:
                paddle.enable_static()
                self.input_x = np.random.rand(5, 100).astype('float64')
                self.input_vec = np.random.rand(100).astype('float64')
                with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                    data_x = paddle.static.data('x', shape=[5, 100], dtype='float64')
                    data_vec = paddle.static.data('vec', shape=[100], dtype='float64')
                    data_x.stop_gradient = x_stop_gradient
                    data_vec.stop_gradient = vec_stop_gradient
                    result_vec = paddle.mv(data_x, data_vec)
                    self.place = paddle.CPUPlace()
                    exe = paddle.static.Executor(self.place)
                    (res,) = exe.run(feed={'x': self.input_x, 'vec': self.input_vec}, fetch_list=[result_vec])
                    z_expected = np.array(np.dot(self.input_x, self.input_vec))
                    np.testing.assert_allclose(res, z_expected, rtol=1e-05)

class TestMVError(unittest.TestCase):

    @test_with_pir_api
    def test_input(self):
        if False:
            i = 10
            return i + 15

        def test_shape():
            if False:
                print('Hello World!')
            paddle.enable_static()
            self.input_x = np.random.rand(5, 100).astype('float64')
            self.input_vec = np.random.rand(100).astype('float64')
            data_x = paddle.static.data('x', shape=[5, 100], dtype='float64')
            data_vec = paddle.static.data('vec', shape=[100, 2], dtype='float64')
            result_vec = paddle.mv(data_x, data_vec)
        self.assertRaises(ValueError, test_shape)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()