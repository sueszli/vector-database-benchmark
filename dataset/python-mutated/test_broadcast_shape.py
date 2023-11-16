import unittest
import paddle

class TestBroadcastShape(unittest.TestCase):

    def test_result(self):
        if False:
            while True:
                i = 10
        shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
        self.assertEqual(shape, [2, 3, 3])
        shape = paddle.broadcast_shape([-1, 1, 3], [1, 3, 1])
        self.assertEqual(shape, [-1, 3, 3])

    def test_error(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, paddle.broadcast_shape, [2, 1, 3], [3, 3, 1])
if __name__ == '__main__':
    unittest.main()