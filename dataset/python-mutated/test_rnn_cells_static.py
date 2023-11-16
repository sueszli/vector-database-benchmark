import paddle
paddle.framework.set_default_dtype('float64')
paddle.enable_static()
import unittest
import numpy as np
from convert import convert_params_for_cell_static
from rnn_numpy import GRUCell, LSTMCell, SimpleRNNCell

class TestSimpleRNNCell(unittest.TestCase):

    def __init__(self, bias=True, place='cpu'):
        if False:
            while True:
                i = 10
        super().__init__(methodName='runTest')
        self.bias = bias
        self.place = paddle.CPUPlace() if place == 'cpu' else paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            print('Hello World!')
        rnn1 = SimpleRNNCell(16, 32, bias=self.bias)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.SimpleRNNCell(16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        place = self.place
        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_cell_static(rnn1, rnn2, place)
        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        if False:
            i = 10
            return i + 15
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        (y1, h1) = rnn1(x, prev_h)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, 16], dtype=paddle.framework.get_default_dtype())
                init_h = paddle.static.data('init_h', [-1, 32], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data, init_h)
        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            print('Hello World!')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(4, 16)
        (y1, h1) = rnn1(x)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, 16], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data)
        feed_dict = {x_data.name: x}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h], use_prune=True)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def runTest(self):
        if False:
            return 10
        self.test_with_initial_state()
        self.test_with_zero_state()

class TestGRUCell(unittest.TestCase):

    def __init__(self, bias=True, place='cpu'):
        if False:
            while True:
                i = 10
        super().__init__(methodName='runTest')
        self.bias = bias
        self.place = paddle.CPUPlace() if place == 'cpu' else paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        rnn1 = GRUCell(16, 32, bias=self.bias)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.GRUCell(16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        place = self.place
        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_cell_static(rnn1, rnn2, place)
        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.place = place
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        if False:
            return 10
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        (y1, h1) = rnn1(x, prev_h)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, 16], dtype=paddle.framework.get_default_dtype())
                init_h = paddle.static.data('init_h', [-1, 32], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data, init_h)
        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            while True:
                i = 10
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(4, 16)
        (y1, h1) = rnn1(x)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, 16], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data)
        feed_dict = {x_data.name: x}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h], use_prune=True)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def runTest(self):
        if False:
            i = 10
            return i + 15
        self.test_with_initial_state()
        self.test_with_zero_state()

class TestLSTMCell(unittest.TestCase):

    def __init__(self, bias=True, place='cpu'):
        if False:
            while True:
                i = 10
        super().__init__(methodName='runTest')
        self.bias = bias
        self.place = paddle.CPUPlace() if place == 'cpu' else paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        rnn1 = LSTMCell(16, 32, bias=self.bias)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTMCell(16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        place = self.place
        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_cell_static(rnn1, rnn2, place)
        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.place = place
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        if False:
            print('Hello World!')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        prev_c = np.random.randn(4, 32)
        (y1, (h1, c1)) = rnn1(x, (prev_h, prev_c))
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, 16], dtype=paddle.framework.get_default_dtype())
                init_h = paddle.static.data('init_h', [-1, 32], dtype=paddle.framework.get_default_dtype())
                init_c = paddle.static.data('init_c', [-1, 32], dtype=paddle.framework.get_default_dtype())
                (y, (h, c)) = rnn2(x_data, (init_h, init_c))
        feed_dict = {x_data.name: x, init_h.name: prev_h, init_c.name: prev_c}
        with paddle.static.scope_guard(scope):
            (y2, h2, c2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2, atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            print('Hello World!')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(4, 16)
        (y1, (h1, c1)) = rnn1(x)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, 16], dtype=paddle.framework.get_default_dtype())
                (y, (h, c)) = rnn2(x_data)
        feed_dict = {x_data.name: x}
        with paddle.static.scope_guard(scope):
            (y2, h2, c2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c], use_prune=True)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2, atol=1e-08, rtol=1e-05)

    def runTest(self):
        if False:
            print('Hello World!')
        self.test_with_initial_state()
        self.test_with_zero_state()

def load_tests(loader, tests, pattern):
    if False:
        for i in range(10):
            print('nop')
    suite = unittest.TestSuite()
    devices = ['cpu', 'gpu'] if paddle.base.is_compiled_with_cuda() else ['cpu']
    for bias in [True, False]:
        for device in devices:
            for test_class in [TestSimpleRNNCell, TestGRUCell, TestLSTMCell]:
                suite.addTest(test_class(bias, device))
    return suite