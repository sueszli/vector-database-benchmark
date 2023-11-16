import unittest
import numpy as np
from test_sum_op import TestReduceOPTensorAxisBase
import paddle

class TestProdOp(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.input = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        if False:
            while True:
                i = 10
        input = paddle.to_tensor(self.input)
        dy_result = paddle.prod(input)
        expected_result = np.prod(self.input)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)
        dy_result = paddle.prod(input, axis=1)
        expected_result = np.prod(self.input, axis=1)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)
        dy_result = paddle.prod(input, axis=-1)
        expected_result = np.prod(self.input, axis=-1)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)
        dy_result = paddle.prod(input, axis=[0, 1])
        expected_result = np.prod(self.input, axis=(0, 1))
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05, atol=1e-08)
        dy_result = paddle.prod(input, axis=1, keepdim=True)
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)
        dy_result = paddle.prod(input, axis=1, dtype='int64')
        expected_result = np.prod(self.input, axis=1, dtype=np.int64)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)
        dy_result = paddle.prod(input, axis=1, keepdim=True, dtype='int64')
        expected_result = np.prod(self.input, axis=1, keepdims=True, dtype=np.int64)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)

    def run_static(self):
        if False:
            print('Hello World!')
        input = paddle.static.data(name='input', shape=[10, 10, 5], dtype='float32')
        result0 = paddle.prod(input)
        result1 = paddle.prod(input, axis=1)
        result2 = paddle.prod(input, axis=-1)
        result3 = paddle.prod(input, axis=[0, 1])
        result4 = paddle.prod(input, axis=1, keepdim=True)
        result5 = paddle.prod(input, axis=1, dtype='int64')
        result6 = paddle.prod(input, axis=1, keepdim=True, dtype='int64')
        place = paddle.XPUPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        static_result = exe.run(feed={'input': self.input}, fetch_list=[result0, result1, result2, result3, result4, result5, result6])
        expected_result = np.prod(self.input)
        np.testing.assert_allclose(static_result[0], expected_result, rtol=1e-05)
        expected_result = np.prod(self.input, axis=1)
        np.testing.assert_allclose(static_result[1], expected_result, rtol=1e-05)
        expected_result = np.prod(self.input, axis=-1)
        np.testing.assert_allclose(static_result[2], expected_result, rtol=1e-05)
        expected_result = np.prod(self.input, axis=(0, 1))
        np.testing.assert_allclose(static_result[3], expected_result, rtol=1e-05, atol=1e-08)
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        np.testing.assert_allclose(static_result[4], expected_result, rtol=1e-05)
        expected_result = np.prod(self.input, axis=1, dtype=np.int64)
        np.testing.assert_allclose(static_result[5], expected_result, rtol=1e-05)
        expected_result = np.prod(self.input, axis=1, keepdims=True, dtype=np.int64)
        np.testing.assert_allclose(static_result[6], expected_result, rtol=1e-05)

    def test_xpu(self):
        if False:
            print('Hello World!')
        paddle.disable_static(place=paddle.XPUPlace(0))
        self.run_imperative()
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static()

class TestProdOpError(unittest.TestCase):

    def test_error(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[2, 2, 4], dtype='float32')
            bool_x = paddle.static.data(name='bool_x', shape=[2, 2, 4], dtype='bool')
            self.assertRaises(TypeError, paddle.prod, [1])
            self.assertRaises(TypeError, paddle.prod, bool_x)
            self.assertRaises(TypeError, paddle.prod, x, 1.5)
            self.assertRaises(TypeError, paddle.prod, x, 'bool')

class TestProdWithTensorAxis1(TestReduceOPTensorAxisBase):

    def init_data(self):
        if False:
            return 10
        self.pd_api = paddle.prod
        self.np_api = np.prod
        self.x = paddle.randn([10, 5, 9, 9], dtype='float32')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')

class TestProdWithTensorAxis2(TestReduceOPTensorAxisBase):

    def init_data(self):
        if False:
            return 10
        self.pd_api = paddle.prod
        self.np_api = np.prod
        self.x = paddle.randn([10, 10, 9, 9], dtype='float32')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [0, paddle.to_tensor([1], 'int64'), paddle.to_tensor([2], 'int64')]
if __name__ == '__main__':
    unittest.main()