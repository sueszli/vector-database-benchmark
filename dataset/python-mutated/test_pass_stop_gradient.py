import unittest
import paddle

class TestStopGradient(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()

    def tearDown(self):
        if False:
            print('Hello World!')
        paddle.disable_static()

    def create_var(self, stop_gradient):
        if False:
            return 10
        x = paddle.randn([2, 4])
        x.stop_gradient = stop_gradient
        return x

    def test_unary(self):
        if False:
            while True:
                i = 10
        x = self.create_var(True)
        out = x.reshape([4, -1])
        self.assertTrue(out.stop_gradient)

    def test_binary(self):
        if False:
            i = 10
            return i + 15
        x = self.create_var(True)
        y = self.create_var(True)
        out = x + y
        self.assertTrue(out.stop_gradient)

    def test_binary2(self):
        if False:
            print('Hello World!')
        x = self.create_var(True)
        y = self.create_var(False)
        out = x + y
        self.assertFalse(out.stop_gradient)
if __name__ == '__main__':
    unittest.main()