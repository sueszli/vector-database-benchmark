import unittest
import numpy as np
import paddle
from paddle.static import Program, program_guard

class TestMultiplyApi(unittest.TestCase):

    def _run_static_graph_case(self, x_data, y_data):
        if False:
            return 10
        with program_guard(Program(), Program()):
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=x_data.shape, dtype=x_data.dtype)
            y = paddle.static.data(name='y', shape=y_data.shape, dtype=y_data.dtype)
            res = paddle.inner(x, y)
            place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            outs = exe.run(paddle.static.default_main_program(), feed={'x': x_data, 'y': y_data}, fetch_list=[res])
            res = outs[0]
            return res

    def _run_dynamic_graph_case(self, x_data, y_data):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        res = paddle.inner(x, y)
        return res.numpy()

    def test_multiply(self):
        if False:
            return 10
        np.random.seed(7)
        x_data = np.random.rand(2, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 5, 10).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(200, 5).astype(np.float64)
        y_data = np.random.rand(50, 5).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(20, 10).astype(np.float32)
        y_data = np.random.rand(1).astype(np.float32).item()
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(20, 50).astype(np.float64) + 1j * np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64) + 1j * np.random.rand(50).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(5, 10, 10).astype(np.float64) + 1j * np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64) + 1j * np.random.rand(2, 10).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

class TestMultiplyError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[100], dtype=np.int8)
            y = paddle.static.data(name='y', shape=[100], dtype=np.int8)
            self.assertRaises(TypeError, paddle.inner, x, y)
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[20, 50], dtype=np.float64)
            y = paddle.static.data(name='y', shape=[20], dtype=np.float64)
            self.assertRaises(ValueError, paddle.inner, x, y)
        np.random.seed(7)
        x_data = np.random.rand(20, 5)
        y_data = np.random.rand(10, 2)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        self.assertRaises(ValueError, paddle.inner, x, y)
        x_data = np.random.randn(200).astype(np.float64)
        y_data = np.random.randn(200).astype(np.float64)
        y = paddle.to_tensor(y_data)
        self.assertRaises(TypeError, paddle.inner, x_data, y)
        x_data = np.random.randn(200).astype(np.float64)
        y_data = np.random.randn(200).astype(np.float64)
        x = paddle.to_tensor(x_data)
        self.assertRaises(TypeError, paddle.inner, x, y_data)
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        self.assertRaises(TypeError, paddle.inner, x_data, y_data)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()