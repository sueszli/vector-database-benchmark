import paddle
paddle.framework.set_default_dtype('float64')
import unittest
import numpy as np
from convert import convert_params_for_cell
from rnn_numpy import GRUCell, LSTMCell, SimpleRNNCell

class TestSimpleRNNCell(unittest.TestCase):

    def __init__(self, weight=True, bias=True, place='cpu'):
        if False:
            print('Hello World!')
        super().__init__(methodName='runTest')
        self.weight = weight
        self.bias = bias
        self.place = paddle.CPUPlace() if place == 'cpu' else paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static(self.place)
        rnn1 = SimpleRNNCell(16, 32, weight=self.weight, bias=self.bias)
        rnn2 = paddle.nn.SimpleRNNCell(16, 32, weight_ih_attr=self.weight, weight_hh_attr=self.weight, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        convert_params_for_cell(rnn1, rnn2)
        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        if False:
            print('Hello World!')
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        (y1, h1) = rnn1(x, prev_h)
        (y2, h2) = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            for i in range(10):
                print('nop')
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        x = np.random.randn(4, 16)
        (y1, h1) = rnn1(x)
        (y2, h2) = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-08, rtol=1e-05)

    def test_errors(self):
        if False:
            return 10

        def test_zero_hidden_size():
            if False:
                print('Hello World!')
            cell = paddle.nn.SimpleRNNCell(-1, 0)
        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        if False:
            while True:
                i = 10
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()

class TestGRUCell(unittest.TestCase):

    def __init__(self, weight=True, bias=True, place='cpu'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(methodName='runTest')
        self.weight = weight
        self.bias = bias
        self.place = paddle.CPUPlace() if place == 'cpu' else paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        rnn1 = GRUCell(16, 32, weight=self.weight, bias=self.bias)
        rnn2 = paddle.nn.GRUCell(16, 32, weight_ih_attr=self.weight, weight_hh_attr=self.weight, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        convert_params_for_cell(rnn1, rnn2)
        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        if False:
            return 10
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        (y1, h1) = rnn1(x, prev_h)
        (y2, h2) = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            print('Hello World!')
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        x = np.random.randn(4, 16)
        (y1, h1) = rnn1(x)
        (y2, h2) = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-08, rtol=1e-05)

    def test_errors(self):
        if False:
            while True:
                i = 10

        def test_zero_hidden_size():
            if False:
                i = 10
                return i + 15
            cell = paddle.nn.GRUCell(-1, 0)
        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        if False:
            return 10
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()

class TestLSTMCell(unittest.TestCase):

    def __init__(self, weight=True, bias=True, place='cpu'):
        if False:
            i = 10
            return i + 15
        super().__init__(methodName='runTest')
        self.weight = weight
        self.bias = bias
        self.place = paddle.CPUPlace() if place == 'cpu' else paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        rnn1 = LSTMCell(16, 32, weight=self.weight, bias=self.bias)
        rnn2 = paddle.nn.LSTMCell(16, 32, weight_ih_attr=self.weight, weight_hh_attr=self.weight, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        convert_params_for_cell(rnn1, rnn2)
        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        if False:
            return 10
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        prev_c = np.random.randn(4, 32)
        (y1, (h1, c1)) = rnn1(x, (prev_h, prev_c))
        (y2, (h2, c2)) = rnn2(paddle.to_tensor(x), (paddle.to_tensor(prev_h), paddle.to_tensor(prev_c)))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2.numpy(), atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            for i in range(10):
                print('nop')
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        x = np.random.randn(4, 16)
        (y1, (h1, c1)) = rnn1(x)
        (y2, (h2, c2)) = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2.numpy(), atol=1e-08, rtol=1e-05)

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')

        def test_zero_hidden_size():
            if False:
                while True:
                    i = 10
            cell = paddle.nn.LSTMCell(-1, 0)
        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        if False:
            return 10
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()

def load_tests(loader, tests, pattern):
    if False:
        for i in range(10):
            print('nop')
    suite = unittest.TestSuite()
    devices = ['cpu', 'gpu'] if paddle.base.is_compiled_with_cuda() else ['cpu']
    for weight in [True, False]:
        for bias in [True, False]:
            for device in devices:
                for test_class in [TestSimpleRNNCell, TestGRUCell, TestLSTMCell]:
                    suite.addTest(test_class(weight, bias, device))
    return suite