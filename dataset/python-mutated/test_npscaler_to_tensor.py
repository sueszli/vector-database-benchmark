import unittest
import numpy as np
import paddle
DTYPE_MAP = {paddle.bool: np.bool_, paddle.int32: np.int32, paddle.int64: np.int64, paddle.float16: np.float16, paddle.float32: np.float32, paddle.float64: np.float64, paddle.complex64: np.complex64}

class NumpyScaler2Tensor(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.dtype = np.float32
        self.x_np = np.array([1], dtype=self.dtype)[0]

    def test_dynamic_scaler2tensor(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        self.assertEqual(DTYPE_MAP[x.dtype], self.dtype)
        self.assertEqual(x.numpy(), self.x_np)
        if self.dtype in [np.bool_]:
            return
        self.assertEqual(len(x.shape), 0)

    def test_static_scaler2tensor(self):
        if False:
            print('Hello World!')
        if self.dtype in [np.float16, np.complex64]:
            return
        paddle.enable_static()
        x = paddle.to_tensor(self.x_np)
        self.assertEqual(DTYPE_MAP[x.dtype], self.dtype)
        if self.dtype in [np.bool_, np.float64]:
            return
        self.assertEqual(len(x.shape), 0)

class NumpyScaler2TensorBool(NumpyScaler2Tensor):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dtype = np.bool_
        self.x_np = np.array([1], dtype=self.dtype)[0]

class NumpyScaler2TensorFloat16(NumpyScaler2Tensor):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16
        self.x_np = np.array([1], dtype=self.dtype)[0]

class NumpyScaler2TensorFloat64(NumpyScaler2Tensor):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float64
        self.x_np = np.array([1], dtype=self.dtype)[0]

class NumpyScaler2TensorInt32(NumpyScaler2Tensor):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int32
        self.x_np = np.array([1], dtype=self.dtype)[0]

class NumpyScaler2TensorInt64(NumpyScaler2Tensor):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int64
        self.x_np = np.array([1], dtype=self.dtype)[0]

class NumpyScaler2TensorComplex64(NumpyScaler2Tensor):

    def setUp(self):
        if False:
            return 10
        self.dtype = np.complex64
        self.x_np = np.array([1], dtype=self.dtype)[0]