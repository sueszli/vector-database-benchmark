import unittest
import numpy as np
import paddle
from paddle import base

class TestFunctionalL1Loss(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        if False:
            i = 10
            return i + 15
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        dy_result = paddle.nn.functional.l1_loss(input, label)
        expected = np.mean(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        dy_result = paddle.nn.functional.l1_loss(input, label, reduction='sum')
        expected = np.sum(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        dy_result = paddle.nn.functional.l1_loss(input, label, reduction='none')
        expected = np.abs(self.input_np - self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [10, 10, 5])

    def run_static(self, use_gpu=False):
        if False:
            i = 10
            return i + 15
        input = paddle.static.data(name='input', shape=[10, 10, 5], dtype='float32')
        label = paddle.static.data(name='label', shape=[10, 10, 5], dtype='float32')
        result0 = paddle.nn.functional.l1_loss(input, label)
        result1 = paddle.nn.functional.l1_loss(input, label, reduction='sum')
        result2 = paddle.nn.functional.l1_loss(input, label, reduction='none')
        y = paddle.nn.functional.l1_loss(input, label, name='aaa')
        place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        static_result = exe.run(feed={'input': self.input_np, 'label': self.label_np}, fetch_list=[result0, result1, result2])
        expected = np.mean(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(static_result[0], expected, rtol=1e-05)
        expected = np.sum(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(static_result[1], expected, rtol=1e-05)
        expected = np.abs(self.input_np - self.label_np)
        np.testing.assert_allclose(static_result[2], expected, rtol=1e-05)
        self.assertTrue('aaa' in y.name)

    def test_cpu(self):
        if False:
            return 10
        paddle.disable_static(place=paddle.base.CPUPlace())
        self.run_imperative()
        paddle.enable_static()
        with base.program_guard(base.Program()):
            self.run_static()

    def test_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        if not base.core.is_compiled_with_cuda():
            return
        paddle.disable_static(place=paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()
        with base.program_guard(base.Program()):
            self.run_static(use_gpu=True)

    def test_errors(self):
        if False:
            i = 10
            return i + 15

        def test_value_error():
            if False:
                print('Hello World!')
            input = paddle.static.data(name='input', shape=[10, 10, 5], dtype='float32')
            label = paddle.static.data(name='label', shape=[10, 10, 5], dtype='float32')
            loss = paddle.nn.functional.l1_loss(input, label, reduction='reduce_mean')
        self.assertRaises(ValueError, test_value_error)

class TestClassL1Loss(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        if False:
            while True:
                i = 10
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        l1_loss = paddle.nn.loss.L1Loss()
        dy_result = l1_loss(input, label)
        expected = np.mean(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
        dy_result = l1_loss(input, label)
        expected = np.sum(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        l1_loss = paddle.nn.loss.L1Loss(reduction='none')
        dy_result = l1_loss(input, label)
        expected = np.abs(self.input_np - self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [10, 10, 5])

    def run_static(self, use_gpu=False):
        if False:
            i = 10
            return i + 15
        input = paddle.static.data(name='input', shape=[10, 10, 5], dtype='float32')
        label = paddle.static.data(name='label', shape=[10, 10, 5], dtype='float32')
        l1_loss = paddle.nn.loss.L1Loss()
        result0 = l1_loss(input, label)
        l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
        result1 = l1_loss(input, label)
        l1_loss = paddle.nn.loss.L1Loss(reduction='none')
        result2 = l1_loss(input, label)
        l1_loss = paddle.nn.loss.L1Loss(name='aaa')
        result3 = l1_loss(input, label)
        place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        static_result = exe.run(feed={'input': self.input_np, 'label': self.label_np}, fetch_list=[result0, result1, result2])
        expected = np.mean(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(static_result[0], expected, rtol=1e-05)
        expected = np.sum(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(static_result[1], expected, rtol=1e-05)
        expected = np.abs(self.input_np - self.label_np)
        np.testing.assert_allclose(static_result[2], expected, rtol=1e-05)
        self.assertTrue('aaa' in result3.name)

    def test_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static(place=paddle.base.CPUPlace())
        self.run_imperative()
        paddle.enable_static()
        with base.program_guard(base.Program()):
            self.run_static()

    def test_gpu(self):
        if False:
            return 10
        if not base.core.is_compiled_with_cuda():
            return
        paddle.disable_static(place=paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()
        with base.program_guard(base.Program()):
            self.run_static(use_gpu=True)

    def test_errors(self):
        if False:
            return 10

        def test_value_error():
            if False:
                for i in range(10):
                    print('nop')
            loss = paddle.nn.loss.L1Loss(reduction='reduce_mean')
        self.assertRaises(ValueError, test_value_error)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()