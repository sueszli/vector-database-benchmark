import paddle
paddle.set_default_dtype('float64')
paddle.enable_static()
import unittest
import numpy as np
from convert import convert_params_for_net_static
from rnn_numpy import GRU, LSTM, SimpleRNN
bidirectional_list = ['bidirectional', 'bidirect']

class TestSimpleRNN(unittest.TestCase):

    def __init__(self, time_major=True, direction='forward', place='cpu'):
        if False:
            i = 10
            return i + 15
        super().__init__('runTest')
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        place = paddle.set_device(self.place)
        rnn1 = SimpleRNN(16, 32, 2, time_major=self.time_major, direction=self.direction)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.SimpleRNN(16, 32, 2, time_major=self.time_major, direction=self.direction)
        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)
        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.place = place
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        if False:
            i = 10
            return i + 15
        mp = self.mp.clone().clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)
        (y1, h1) = rnn1(x, prev_h)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                init_h = paddle.static.data('init_h', [2 * self.num_directions, -1, 32], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data, init_h)
        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            for i in range(10):
                print('nop')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        (y1, h1) = rnn1(x)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data)
        feed_dict = {x_data.name: x}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def test_with_input_lengths(self):
        if False:
            print('Hello World!')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)
        (y1, h1) = rnn1(x, sequence_length=sequence_length)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                seq_len = paddle.static.data('seq_len', [-1], dtype='int64')
                mask = paddle.static.nn.sequence_lod.sequence_mask(seq_len, dtype=paddle.get_default_dtype())
                if self.time_major:
                    mask = paddle.transpose(mask, [1, 0])
                (y, h) = rnn2(x_data, sequence_length=seq_len)
                mask = paddle.unsqueeze(mask, -1)
                y = paddle.multiply(y, mask)
        feed_dict = {x_data.name: x, seq_len.name: sequence_length}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def runTest(self):
        if False:
            i = 10
            return i + 15
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_with_input_lengths()

class TestGRU(unittest.TestCase):

    def __init__(self, time_major=True, direction='forward', place='cpu'):
        if False:
            return 10
        super().__init__('runTest')
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        if False:
            while True:
                i = 10
        place = paddle.set_device(self.place)
        rnn1 = GRU(16, 32, 2, time_major=self.time_major, direction=self.direction)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.GRU(16, 32, 2, time_major=self.time_major, direction=self.direction)
        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)
        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.place = place
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
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)
        (y1, h1) = rnn1(x, prev_h)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                init_h = paddle.static.data('init_h', [2 * self.num_directions, -1, 32], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data, init_h)
        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
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
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        (y1, h1) = rnn1(x)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                (y, h) = rnn2(x_data)
        feed_dict = {x_data.name: x}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def test_with_input_lengths(self):
        if False:
            print('Hello World!')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)
        (y1, h1) = rnn1(x, sequence_length=sequence_length)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                seq_len = paddle.static.data('seq_len', [-1], dtype='int64')
                mask = paddle.static.nn.sequence_lod.sequence_mask(seq_len, dtype=paddle.get_default_dtype())
                if self.time_major:
                    mask = paddle.transpose(mask, [1, 0])
                (y, h) = rnn2(x_data, sequence_length=seq_len)
                mask = paddle.unsqueeze(mask, -1)
                y = paddle.multiply(y, mask)
        feed_dict = {x_data.name: x, seq_len.name: sequence_length}
        with paddle.static.scope_guard(scope):
            (y2, h2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)

    def runTest(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_with_initial_state()
        self.test_with_zero_state()

class TestLSTM(unittest.TestCase):

    def __init__(self, time_major=True, direction='forward', place='cpu'):
        if False:
            return 10
        super().__init__('runTest')
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        if False:
            while True:
                i = 10
        place = paddle.set_device(self.place)
        rnn1 = LSTM(16, 32, 2, time_major=self.time_major, direction=self.direction)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTM(16, 32, 2, time_major=self.time_major, direction=self.direction)
        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)
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
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)
        prev_c = np.random.randn(2 * self.num_directions, 4, 32)
        (y1, (h1, c1)) = rnn1(x, (prev_h, prev_c))
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                init_h = paddle.static.data('init_h', [2 * self.num_directions, -1, 32], dtype=paddle.framework.get_default_dtype())
                init_c = paddle.static.data('init_c', [2 * self.num_directions, -1, 32], dtype=paddle.framework.get_default_dtype())
                (y, (h, c)) = rnn2(x_data, (init_h, init_c))
        feed_dict = {x_data.name: x, init_h.name: prev_h, init_c.name: prev_c}
        with paddle.static.scope_guard(scope):
            (y2, h2, c2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2, atol=1e-08, rtol=1e-05)

    def test_with_zero_state(self):
        if False:
            return 10
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        (y1, (h1, c1)) = rnn1(x)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                (y, (h, c)) = rnn2(x_data)
        feed_dict = {x_data.name: x}
        with paddle.static.scope_guard(scope):
            (y2, h2, c2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2, atol=1e-08, rtol=1e-05)

    def test_with_input_lengths(self):
        if False:
            for i in range(10):
                print('nop')
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)
        (y1, (h1, c1)) = rnn1(x, sequence_length=sequence_length)
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data('input', [-1, -1, 16], dtype=paddle.framework.get_default_dtype())
                seq_len = paddle.static.data('seq_len', [-1], dtype='int64')
                mask = paddle.static.nn.sequence_lod.sequence_mask(seq_len, dtype=paddle.get_default_dtype())
                if self.time_major:
                    mask = paddle.transpose(mask, [1, 0])
                (y, (h, c)) = rnn2(x_data, sequence_length=seq_len)
                mask = paddle.unsqueeze(mask, -1)
                y = paddle.multiply(y, mask)
        feed_dict = {x_data.name: x, seq_len.name: sequence_length}
        with paddle.static.scope_guard(scope):
            (y2, h2, c2) = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])
        np.testing.assert_allclose(y1, y2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(h1, h2, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(c1, c2, atol=1e-08, rtol=1e-05)

    def runTest(self):
        if False:
            print('Hello World!')
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_with_input_lengths()

def load_tests(loader, tests, pattern):
    if False:
        print('Hello World!')
    suite = unittest.TestSuite()
    devices = ['cpu', 'gpu'] if paddle.base.is_compiled_with_cuda() else ['cpu']
    for direction in ['forward', 'bidirectional', 'bidirect']:
        for time_major in [True, False]:
            for device in devices:
                for test_class in [TestSimpleRNN, TestLSTM, TestGRU]:
                    suite.addTest(test_class(time_major, direction, device))
    return suite
if __name__ == '__main__':
    unittest.main()