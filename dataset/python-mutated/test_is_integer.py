import unittest
import numpy as np
import paddle

class TestIsInteger(unittest.TestCase):

    def test_for_integer(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.arange(10)
        self.assertTrue(paddle.is_integer(x))

    def test_for_floating_point(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.randn([2, 3])
        self.assertFalse(paddle.is_integer(x))

    def test_for_complex(self):
        if False:
            while True:
                i = 10
        x = paddle.randn([2, 3]) + 1j * paddle.randn([2, 3])
        self.assertFalse(paddle.is_integer(x))

    def test_for_exception(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            paddle.is_integer(np.array([1, 2]))
if __name__ == '__main__':
    unittest.main()