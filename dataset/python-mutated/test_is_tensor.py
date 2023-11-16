import unittest
import paddle
DELTA = 1e-05

class TestIsTensorApi(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.disable_static()

    def tearDown(self):
        if False:
            return 10
        paddle.enable_static()

    def test_is_tensor_real(self, dtype='float32'):
        if False:
            return 10
        'Test is_tensor api with a real tensor'
        x = paddle.rand([3, 2, 4], dtype=dtype)
        self.assertTrue(paddle.is_tensor(x))

    def test_is_tensor_list(self, dtype='float32'):
        if False:
            for i in range(10):
                print('nop')
        'Test is_tensor api with a list'
        x = [1, 2, 3]
        self.assertFalse(paddle.is_tensor(x))

    def test_is_tensor_number(self, dtype='float32'):
        if False:
            while True:
                i = 10
        'Test is_tensor api with a number'
        x = 5
        self.assertFalse(paddle.is_tensor(x))

class TestIsTensorStatic(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()

    def test_is_tensor(self):
        if False:
            print('Hello World!')
        x = paddle.rand([3, 2, 4], dtype='float32')
        self.assertTrue(paddle.is_tensor(x))

    def test_is_tensor_array(self):
        if False:
            print('Hello World!')
        x = paddle.tensor.create_array('float32')
        self.assertTrue(paddle.is_tensor(x))
if __name__ == '__main__':
    unittest.main()