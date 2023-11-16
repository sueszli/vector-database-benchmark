import unittest
import paddle

class TestSundryAPI(unittest.TestCase):

    def test_detach(self):
        if False:
            return 10
        x = paddle.rand([0, 2])
        out = x.detach()
        self.assertEqual(out.shape, [0, 2])
        self.assertEqual(out.size, 0)

    def test_numpy(self):
        if False:
            return 10
        x = paddle.rand([0, 2])
        out = x.numpy()
        self.assertEqual(out.shape, (0, 2))
        self.assertEqual(out.size, 0)

    def test_reshape(self):
        if False:
            while True:
                i = 10
        x1 = paddle.rand([0, 2])
        x1.stop_gradient = False
        out1 = paddle.reshape(x1, [-1])
        self.assertEqual(out1.shape, [0])
        self.assertEqual(out1.size, 0)
        x2 = paddle.rand([0, 2])
        x2.stop_gradient = False
        out2 = paddle.reshape(x2, [2, -1])
        self.assertEqual(out2.shape, [2, 0])
        self.assertEqual(out2.size, 0)
        x3 = paddle.rand([0, 2])
        x3.stop_gradient = False
        out3 = paddle.reshape(x3, [2, 3, 0])
        self.assertEqual(out3.shape, [2, 3, 0])
        self.assertEqual(out3.size, 0)
        x4 = paddle.rand([0, 2])
        x4.stop_gradient = False
        out4 = paddle.reshape(x4, [0])
        self.assertEqual(out4.shape, [0])
        self.assertEqual(out4.size, 0)
        x5 = paddle.rand([0])
        with self.assertRaises(ValueError):
            out4 = paddle.reshape(x5, [2, 0, -1])
if __name__ == '__main__':
    unittest.main()