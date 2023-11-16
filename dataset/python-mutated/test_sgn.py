import unittest
import numpy as np
import paddle

def np_sgn(x: np.ndarray):
    if False:
        for i in range(10):
            print('nop')
    if x.dtype == 'complex128' or x.dtype == 'complex64':
        x_abs = np.abs(x)
        eps = np.finfo(x.dtype).eps
        x_abs = np.maximum(x_abs, eps)
        out = x / x_abs
    else:
        out = np.sign(x)
    return out

class TestSgnError(unittest.TestCase):

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        input2 = paddle.to_tensor(np.random.randint(-10, 10, size=[12, 20]).astype('int32'))
        input3 = paddle.to_tensor(np.random.randint(-10, 10, size=[12, 20]).astype('int64'))
        self.assertRaises(TypeError, paddle.sgn, input2)
        self.assertRaises(TypeError, paddle.sgn, input3)

class TestSignAPI(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.support_dtypes = ['float16', 'float32', 'float64', 'complex64', 'complex128']
        if paddle.device.get_device() == 'cpu':
            self.support_dtypes = ['float32', 'float64', 'complex64', 'complex128']

    def test_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.support_dtypes:
            x = paddle.to_tensor(np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype))
            paddle.sgn(x)

    def test_complex(self):
        if False:
            return 10
        for dtype in ['complex64', 'complex128']:
            np_x = np.array([[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]], dtype=dtype)
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)

    def test_float(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.support_dtypes:
            np_x = np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype)
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()