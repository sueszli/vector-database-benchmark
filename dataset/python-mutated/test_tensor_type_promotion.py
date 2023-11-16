import unittest
import warnings
import paddle

class TestTensorTypePromotion(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = paddle.to_tensor([2, 3])
        self.y = paddle.to_tensor([1.0, 2.0])

    def add_operator(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter('always')
            self.x + self.y

    def sub_operator(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter('always')
            self.x - self.y

    def mul_operator(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter('always')
            self.x * self.y

    def div_operator(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter('always')
            self.x / self.y

    def test_operator(self):
        if False:
            i = 10
            return i + 15
        self.setUp()
        self.add_operator()
        self.sub_operator()
        self.mul_operator()
        self.div_operator()
if __name__ == '__main__':
    unittest.main()