import unittest
import numpy as np
from paddle import to_tensor
from paddle.nn import ZeroPad2D
from paddle.nn.functional import zeropad2d

class TestZeroPad2dAPIError(unittest.TestCase):
    """
    test paddle.zeropad2d error.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        unsupport dtypes\n        '
        self.shape = [4, 3, 224, 224]
        self.unsupport_dtypes = ['bool', 'int8']

    def test_unsupport_dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test unsupport dtypes.\n        '
        for dtype in self.unsupport_dtypes:
            pad = 2
            x = np.random.randint(-255, 255, size=self.shape)
            x_tensor = to_tensor(x).astype(dtype)
            self.assertRaises(TypeError, zeropad2d, x=x_tensor, padding=pad)

class TestZeroPad2dAPI(unittest.TestCase):
    """
    test paddle.zeropad2d
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        support dtypes\n        '
        self.shape = [4, 3, 224, 224]
        self.support_dtypes = ['float32', 'float64', 'int32', 'int64']

    def test_support_dtypes(self):
        if False:
            print('Hello World!')
        '\n        test support types\n        '
        for dtype in self.support_dtypes:
            pad = 2
            x = np.random.randint(-255, 255, size=self.shape).astype(dtype)
            expect_res = np.pad(x, [[0, 0], [0, 0], [pad, pad], [pad, pad]])
            x_tensor = to_tensor(x).astype(dtype)
            ret_res = zeropad2d(x_tensor, [pad, pad, pad, pad]).numpy()
            np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_support_pad2(self):
        if False:
            while True:
                i = 10
        "\n        test the type of 'pad' is list.\n        "
        pad = [1, 2, 3, 4]
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])
        x_tensor = to_tensor(x)
        ret_res = zeropad2d(x_tensor, pad).numpy()
        np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_support_pad3(self):
        if False:
            print('Hello World!')
        "\n        test the type of 'pad' is tuple.\n        "
        pad = (1, 2, 3, 4)
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])
        x_tensor = to_tensor(x)
        ret_res = zeropad2d(x_tensor, pad).numpy()
        np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_support_pad4(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        test the type of 'pad' is paddle.Tensor.\n        "
        pad = [1, 2, 3, 4]
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])
        x_tensor = to_tensor(x)
        pad_tensor = to_tensor(pad, dtype='int32')
        ret_res = zeropad2d(x_tensor, pad_tensor).numpy()
        np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

class TestZeroPad2DLayer(unittest.TestCase):
    """
    test nn.ZeroPad2D
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [4, 3, 224, 224]
        self.pad = [2, 2, 4, 1]
        self.padLayer = ZeroPad2D(padding=self.pad)
        self.x = np.random.randint(-255, 255, size=self.shape)
        self.expect_res = np.pad(self.x, [[0, 0], [0, 0], [self.pad[2], self.pad[3]], [self.pad[0], self.pad[1]]])

    def test_layer(self):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_allclose(zeropad2d(to_tensor(self.x), self.pad).numpy(), self.padLayer(to_tensor(self.x)), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()