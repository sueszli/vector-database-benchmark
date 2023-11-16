import random
import unittest
from contextlib import contextmanager
import numpy as np
from op_test import OpTest, paddle_static_guard
import paddle
from paddle import base

class TestElementwiseModOp(OpTest):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'elementwise_floordiv'
        self.python_api = paddle.floor_divide
        self.dtype = np.int32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_pir=True)

    def init_input_output(self):
        if False:
            return 10
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

    def init_dtype(self):
        if False:
            print('Hello World!')
        pass

    def init_axis(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestElementwiseFloorDivOp_ZeroDim1(TestElementwiseModOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(0, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, []).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

class TestElementwiseFloorDivOp_ZeroDim2(TestElementwiseModOp):

    def init_input_output(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, []).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

class TestElementwiseFloorDivOp_ZeroDim3(TestElementwiseModOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(0, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

class TestElementwiseModOp_scalar(TestElementwiseModOp):

    def init_input_output(self):
        if False:
            return 10
        scale_x = random.randint(0, 100000000)
        scale_y = random.randint(1, 100000000)
        self.x = (np.random.rand(2, 3, 4) * scale_x).astype(self.dtype)
        self.y = (np.random.rand(1) * scale_y + 1).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

class TestElementwiseModOpInverse(TestElementwiseModOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.uniform(0, 10000, [10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

@contextmanager
def device_guard(device=None):
    if False:
        print('Hello World!')
    old = paddle.get_device()
    yield paddle.set_device(device)
    paddle.set_device(old)

class TestFloorDivideOp(unittest.TestCase):

    def test_name(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with paddle_static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='int64')
                y = paddle.static.data(name='y', shape=[2, 3], dtype='int64')
                y_1 = paddle.floor_divide(x, y, name='div_res')
                self.assertEqual('div_res' in y_1.name, True)
            paddle.disable_static()

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            for dtype in ('uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.floor_divide(x, y)
                np_z = z.numpy()
                z_expected = np.floor_divide(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)
            np_x = np.array([2, 3, 8, 7])
            np_y = np.array([1, 5, 3, 3])
            x = paddle.to_tensor(np_x, dtype='bfloat16')
            y = paddle.to_tensor(np_y, dtype='bfloat16')
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([16384, 0, 16384, 16384], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)
            for dtype in ('int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'):
                np_x = -np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.floor_divide(x, y)
                np_z = z.numpy()
                z_expected = np.floor_divide(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)
            np_x = -np.array([2, 3, 8, 7])
            np_y = np.array([1, 5, 3, 3])
            x = paddle.to_tensor(np_x, dtype='bfloat16')
            y = paddle.to_tensor(np_y, dtype='bfloat16')
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([49152, 49024, 49216, 49216], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)
            for dtype in ('float32', 'float64', 'float16'):
                try:
                    np_x = np.array([2])
                    np_y = np.array([0, 0, 0])
                    x = paddle.to_tensor(np_x, dtype=dtype)
                    y = paddle.to_tensor(np_y, dtype=dtype)
                    z = paddle.floor_divide(x, y)
                    np_z = z.numpy()
                    z_expected = np.floor_divide(np_x, np_y)
                    self.assertEqual((np_z == z_expected).all(), True)
                except Exception as e:
                    pass
            np_x = np.array([2])
            np_y = np.array([0, 0, 0])
            x = paddle.to_tensor(np_x, dtype='bfloat16')
            y = paddle.to_tensor(np_y, dtype='bfloat16')
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([32640, 32640, 32640], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)
        with device_guard('cpu'):
            np_x = np.array([2, 3, 4])
            np_y = np.array([0])
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            try:
                z = x // y
            except Exception as e:
                pass
            for dtype in ('uint8', 'int8', 'int16', 'int32', 'int64'):
                np_x = np.array([2])
                np_y = np.array([0, 0, 0])
                x = paddle.to_tensor(np_x, dtype=dtype)
                y = paddle.to_tensor(np_y, dtype=dtype)
                try:
                    z = x // y
                except Exception as e:
                    pass
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()