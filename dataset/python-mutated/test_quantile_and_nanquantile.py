import unittest
import numpy as np
import paddle
API_list = [(paddle.quantile, np.quantile), (paddle.nanquantile, np.nanquantile)]

class TestQuantileAndNanquantile(unittest.TestCase):
    """
    This class is used for numerical precision testing. If there is
    a corresponding numpy API, the precision comparison can be performed directly.
    Otherwise, it needs to be verified by numpy implementated function.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input_data = np.random.rand(4, 7, 6)

    def test_single_q(self):
        if False:
            return 10
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.5, axis=2)
            np_res = res_func(inp, q=0.5, axis=2)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 1, 2] = np.nan

    def test_with_no_axis(self):
        if False:
            i = 10
            return i + 15
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.35)
            np_res = res_func(inp, q=0.35)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 2, 1] = np.nan
            inp[0, 1, 2] = np.nan

    def test_with_multi_axis(self):
        if False:
            while True:
                i = 10
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.75, axis=[0, 2])
            np_res = res_func(inp, q=0.75, axis=[0, 2])
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 5, 3] = np.nan
            inp[0, 6, 2] = np.nan

    def test_with_keepdim(self):
        if False:
            return 10
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.35, axis=2, keepdim=True)
            np_res = res_func(inp, q=0.35, axis=2, keepdims=True)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 3, 4] = np.nan

    def test_with_keepdim_and_multiple_axis(self):
        if False:
            i = 10
            return i + 15
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.1, axis=[1, 2], keepdim=True)
            np_res = res_func(inp, q=0.1, axis=[1, 2], keepdims=True)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 6, 3] = np.nan

    def test_with_boundary_q(self):
        if False:
            while True:
                i = 10
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0, axis=1)
            np_res = res_func(inp, q=0, axis=1)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 2, 5] = np.nan

    def test_quantile_include_NaN(self):
        if False:
            i = 10
            return i + 15
        input_data = np.random.randn(2, 3, 4)
        input_data[0, 1, 1] = np.nan
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.quantile(x, q=0.35, axis=0)
        np_res = np.quantile(input_data, q=0.35, axis=0)
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05, equal_nan=True)

    def test_nanquantile_all_NaN(self):
        if False:
            print('Hello World!')
        input_data = np.full(shape=[2, 3], fill_value=np.nan)
        input_data[0, 2] = 0
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.nanquantile(x, q=0.35, axis=0)
        np_res = np.nanquantile(input_data, q=0.35, axis=0)
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05, equal_nan=True)

class TestMuitlpleQ(unittest.TestCase):
    """
    This class is used to test multiple input of q.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.input_data = np.random.rand(5, 3, 4)

    def test_quantile(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.3, 0.44], axis=-2)
        np_res = np.quantile(self.input_data, q=[0.3, 0.44], axis=-2)
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_quantile_multiple_axis(self):
        if False:
            print('Hello World!')
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.2, 0.67], axis=[1, -1])
        np_res = np.quantile(self.input_data, q=[0.2, 0.67], axis=[1, -1])
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_quantile_multiple_axis_keepdim(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.1, 0.2, 0.3], axis=[1, 2], keepdim=True)
        np_res = np.quantile(self.input_data, q=[0.1, 0.2, 0.3], axis=[1, 2], keepdims=True)
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

class TestError(unittest.TestCase):
    """
    This class is used to test that exceptions are thrown correctly.
    Validity of all parameter values and types should be considered.
    """

    def setUp(self):
        if False:
            return 10
        self.x = paddle.randn((2, 3, 4))

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')

        def test_q_range_error_1():
            if False:
                print('Hello World!')
            paddle_res = paddle.quantile(self.x, q=1.5)
        self.assertRaises(ValueError, test_q_range_error_1)

        def test_q_range_error_2():
            if False:
                for i in range(10):
                    print('nop')
            paddle_res = paddle.quantile(self.x, q=[0.2, -0.3])
        self.assertRaises(ValueError, test_q_range_error_2)

        def test_q_range_error_3():
            if False:
                for i in range(10):
                    print('nop')
            paddle_res = paddle.quantile(self.x, q=[])
        self.assertRaises(ValueError, test_q_range_error_3)

        def test_x_type_error():
            if False:
                while True:
                    i = 10
            x = [1, 3, 4]
            paddle_res = paddle.quantile(x, q=0.9)
        self.assertRaises(TypeError, test_x_type_error)

        def test_axis_type_error_1():
            if False:
                while True:
                    i = 10
            paddle_res = paddle.quantile(self.x, q=0.4, axis=0.4)
        self.assertRaises(ValueError, test_axis_type_error_1)

        def test_axis_type_error_2():
            if False:
                while True:
                    i = 10
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, 0.4])
        self.assertRaises(ValueError, test_axis_type_error_2)

        def test_axis_value_error_1():
            if False:
                while True:
                    i = 10
            paddle_res = paddle.quantile(self.x, q=0.4, axis=10)
        self.assertRaises(ValueError, test_axis_value_error_1)

        def test_axis_value_error_2():
            if False:
                while True:
                    i = 10
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, -10])
        self.assertRaises(ValueError, test_axis_value_error_2)

class TestQuantileRuntime(unittest.TestCase):
    """
    This class is used to test the API could run correctly with
    different devices, different data types, and dygraph/static graph mode.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_data = np.random.rand(4, 7)
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        if paddle.device.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_dygraph(self):
        if False:
            return 10
        paddle.disable_static()
        for (func, res_func) in API_list:
            for device in self.devices:
                paddle.set_device(device)
                for dtype in self.dtypes:
                    np_input_data = self.input_data.astype(dtype)
                    x = paddle.to_tensor(np_input_data, dtype=dtype)
                    paddle_res = func(x, q=0.5, axis=1)
                    np_res = res_func(np_input_data, q=0.5, axis=1)
                    np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_static(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        for (func, res_func) in API_list:
            for device in self.devices:
                x = paddle.static.data(name='x', shape=self.input_data.shape, dtype=paddle.float32)
                x_fp64 = paddle.static.data(name='x_fp64', shape=self.input_data.shape, dtype=paddle.float64)
                results = func(x, q=0.5, axis=1)
                np_input_data = self.input_data.astype('float32')
                results_fp64 = func(x_fp64, q=0.5, axis=1)
                np_input_data_fp64 = self.input_data.astype('float64')
                exe = paddle.static.Executor(device)
                (paddle_res, paddle_res_fp64) = exe.run(paddle.static.default_main_program(), feed={'x': np_input_data, 'x_fp64': np_input_data_fp64}, fetch_list=[results, results_fp64])
                np_res = res_func(np_input_data, q=0.5, axis=1)
                np_res_fp64 = res_func(np_input_data_fp64, q=0.5, axis=1)
                self.assertTrue(np.allclose(paddle_res, np_res) and np.allclose(paddle_res_fp64, np_res_fp64))
if __name__ == '__main__':
    unittest.main()