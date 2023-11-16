import unittest
import numpy as np
from op_test import check_out_dtype
from test_sum_op import TestReduceOPTensorAxisBase
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class ApiMinTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    @test_with_pir_api
    def test_api(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            data = paddle.static.data('data', shape=[10, 10], dtype='float32')
            result_min = paddle.min(x=data, axis=1)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.rand(10, 10).astype(np.float32)
            (res,) = exe.run(feed={'data': input_data}, fetch_list=[result_min])
        self.assertEqual((res == np.min(input_data, axis=1)).all(), True)
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            data = paddle.static.data('data', shape=[10, 10], dtype='int64')
            result_min = paddle.min(x=data, axis=0)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            (res,) = exe.run(feed={'data': input_data}, fetch_list=[result_min])
        self.assertEqual((res == np.min(input_data, axis=0)).all(), True)
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            data = paddle.static.data('data', shape=[10, 10], dtype='int64')
            result_min = paddle.min(x=data, axis=(0, 1))
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            (res,) = exe.run(feed={'data': input_data}, fetch_list=[result_min])
        self.assertEqual((res == np.min(input_data, axis=(0, 1))).all(), True)

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

        def test_input_type():
            if False:
                for i in range(10):
                    print('nop')
            with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                data = np.random.rand(10, 10)
                result_min = paddle.min(x=data, axis=0)
        self.assertRaises(TypeError, test_input_type)

    def test_imperative_api(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        np_x = np.array([10, 10]).astype('float64')
        x = paddle.to_tensor(np_x)
        z = paddle.min(x, axis=0)
        np_z = z.numpy()
        z_expected = np.array(np.min(np_x, axis=0))
        self.assertEqual((np_z == z_expected).all(), True)

    def test_support_tuple(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        np_x = np.array([10, 10]).astype('float64')
        x = paddle.to_tensor(np_x)
        z = paddle.min(x, axis=(0,))
        np_z = z.numpy()
        z_expected = np.array(np.min(np_x, axis=0))
        self.assertEqual((np_z == z_expected).all(), True)

class TestOutDtype(unittest.TestCase):

    def test_min(self):
        if False:
            for i in range(10):
                print('nop')
        api_fn = paddle.min
        shape = [10, 16]
        check_out_dtype(api_fn, in_specs=[(shape,)], expect_dtypes=['float32', 'float64', 'int32', 'int64'])

class TestMinWithTensorAxis1(TestReduceOPTensorAxisBase):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.pd_api = paddle.min
        self.np_api = np.min
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')

class TestMinWithTensorAxis2(TestReduceOPTensorAxisBase):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.pd_api = paddle.min
        self.np_api = np.min
        self.x = paddle.randn([10, 10, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [0, paddle.to_tensor([1], 'int64'), paddle.to_tensor([2], 'int64')]
        self.keepdim = True

class TestMinAPIWithEmptyTensor(unittest.TestCase):

    def test_empty_tensor(self):
        if False:
            return 10
        with base.dygraph.guard():
            with self.assertRaises(ValueError):
                data = np.array([], dtype=np.float32)
                data = np.reshape(data, [0, 0, 0, 0, 0, 0, 0])
                x = paddle.to_tensor(data, dtype='float64')
                np_axis = np.array([0], dtype='int64')
                tensor_axis = paddle.to_tensor(np_axis, dtype='int64')
                out = paddle.min(x, tensor_axis)
if __name__ == '__main__':
    unittest.main()