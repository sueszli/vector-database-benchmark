import unittest
import numpy as np
import paddle
from paddle.pir_utils import test_with_pir_api

class TestMultiplyApi(unittest.TestCase):

    def _run_static_graph_case(self, x_data, y_data):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=x_data.shape, dtype=x_data.dtype)
            y = paddle.static.data(name='y', shape=y_data.shape, dtype=y_data.dtype)
            res = paddle.outer(x, y)
            place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            outs = exe.run(paddle.static.default_main_program(), feed={'x': x_data, 'y': y_data}, fetch_list=[res])
            res = outs[0]
            return res

    def _run_dynamic_graph_case(self, x_data, y_data):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        res = paddle.outer(x, y)
        return res.numpy()

    @test_with_pir_api
    def test_multiply_static(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(7)
        x_data = np.random.rand(2, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 5, 10).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(200, 5).astype(np.float64)
        y_data = np.random.rand(50, 5).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(50).astype(np.int32)
        y_data = np.random.rand(50).astype(np.int32)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(50).astype(np.int64)
        y_data = np.random.rand(50).astype(np.int64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic(self):
        if False:
            print('Hello World!')
        x_data = np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(20, 10).astype(np.float32)
        y_data = np.random.rand(1).astype(np.float32).item()
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=10000.0)
        x_data = np.random.rand(20, 50).astype(np.float64) + 1j * np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64) + 1j * np.random.rand(50).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(5, 10, 10).astype(np.float64) + 1j * np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64) + 1j * np.random.rand(2, 10).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(5, 10, 10).astype(np.int32)
        y_data = np.random.rand(2, 10).astype(np.int32)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)
        x_data = np.random.rand(5, 10, 10).astype(np.int64)
        y_data = np.random.rand(2, 10).astype(np.int64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

class TestMultiplyError(unittest.TestCase):

    def test_errors_static(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[100], dtype=np.int8)
            y = paddle.static.data(name='y', shape=[100], dtype=np.int8)
            self.assertRaises(TypeError, paddle.outer, x, y)

    def test_errors_dynamic(self):
        if False:
            print('Hello World!')
        np.random.seed(7)
        x_data = np.random.randn(200).astype(np.float64)
        y_data = np.random.randn(200).astype(np.float64)
        y = paddle.to_tensor(y_data)
        self.assertRaises(TypeError, paddle.outer, x_data, y)
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        x = paddle.to_tensor(x_data)
        self.assertRaises(TypeError, paddle.outer, x, y_data)
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        self.assertRaises(TypeError, paddle.outer, x_data, y_data)
if __name__ == '__main__':
    unittest.main()