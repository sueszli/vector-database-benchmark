import unittest
import numpy as np
import paddle

class TestIsComplex(unittest.TestCase):

    def test_for_integer(self):
        if False:
            while True:
                i = 10
        x = paddle.arange(10)
        self.assertFalse(paddle.is_complex(x))

    def test_for_floating_point(self):
        if False:
            i = 10
            return i + 15
        x = paddle.randn([2, 3])
        self.assertFalse(paddle.is_complex(x))

    def test_for_complex(self):
        if False:
            print('Hello World!')
        x = paddle.randn([2, 3]) + 1j * paddle.randn([2, 3])
        self.assertTrue(paddle.is_complex(x))

    def test_for_exception(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            paddle.is_complex(np.array([1, 2]))
if __name__ == '__main__':
    unittest.main()