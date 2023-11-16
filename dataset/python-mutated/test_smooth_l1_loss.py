import unittest
import numpy as np
import paddle
from paddle import base

def smooth_l1_loss_forward(val, delta):
    if False:
        print('Hello World!')
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)

def smooth_l1_loss_np(input, label, reduction='mean', delta=1.0):
    if False:
        for i in range(10):
            print('nop')
    diff = input - label
    out = np.vectorize(smooth_l1_loss_forward)(diff, delta)
    if reduction == 'sum':
        return np.sum(out)
    elif reduction == 'mean':
        return np.mean(out)
    elif reduction == 'none':
        return out

class SmoothL1Loss(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(123)

    def test_smooth_l1_loss_mean(self):
        if False:
            return 10
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)
        prog = base.Program()
        startup_prog = base.Program()
        place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(name='input', shape=[100, 200], dtype='float32')
            label = paddle.static.data(name='label', shape=[100, 200], dtype='float32')
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss()
            ret = smooth_l1_loss(input, label)
            exe = base.Executor(place)
            (static_ret,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss()
            dy_ret = smooth_l1_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = smooth_l1_loss_np(input_np, label_np, reduction='mean')
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_smooth_l1_loss_sum(self):
        if False:
            print('Hello World!')
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)
        prog = base.Program()
        startup_prog = base.Program()
        place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(name='input', shape=[100, 200], dtype='float32')
            label = paddle.static.data(name='label', shape=[100, 200], dtype='float32')
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='sum')
            ret = smooth_l1_loss(input, label)
            exe = base.Executor(place)
            (static_ret,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='sum')
            dy_ret = smooth_l1_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = smooth_l1_loss_np(input_np, label_np, reduction='sum')
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_smooth_l1_loss_none(self):
        if False:
            print('Hello World!')
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)
        prog = base.Program()
        startup_prog = base.Program()
        place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(name='input', shape=[100, 200], dtype='float32')
            label = paddle.static.data(name='label', shape=[100, 200], dtype='float32')
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='none')
            ret = smooth_l1_loss(input, label)
            exe = base.Executor(place)
            (static_ret,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='none')
            dy_ret = smooth_l1_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = smooth_l1_loss_np(input_np, label_np, reduction='none')
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_smooth_l1_loss_delta(self):
        if False:
            for i in range(10):
                print('nop')
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)
        delta = np.random.rand()
        prog = base.Program()
        startup_prog = base.Program()
        place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(name='input', shape=[100, 200], dtype='float32')
            label = paddle.static.data(name='label', shape=[100, 200], dtype='float32')
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(delta=delta)
            ret = smooth_l1_loss(input, label)
            exe = base.Executor(place)
            (static_ret,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(delta=delta)
            dy_ret = smooth_l1_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = smooth_l1_loss_np(input_np, label_np, delta=delta)
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()