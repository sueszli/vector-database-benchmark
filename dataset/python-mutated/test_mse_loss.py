import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.base.executor import Executor
from paddle.pir_utils import test_with_pir_api

class TestMseLoss(unittest.TestCase):

    @test_with_pir_api
    def test_mse_loss(self):
        if False:
            for i in range(10):
                print('nop')
        input_val = np.random.uniform(0.1, 0.5, (2, 3)).astype('float32')
        label_val = np.random.uniform(0.1, 0.5, (2, 3)).astype('float32')
        sub = input_val - label_val
        np_result = np.mean(sub * sub)
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input_var = paddle.static.data(name='input', shape=[-1, 3], dtype='float32')
            label_var = paddle.static.data(name='label', shape=[-1, 3], dtype='float32')
            output = paddle.nn.functional.mse_loss(input=input_var, label=label_var)
            for use_cuda in [False, True] if core.is_compiled_with_cuda() else [False]:
                place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                exe = Executor(place)
                (result,) = exe.run(main, feed={'input': input_val, 'label': label_val}, fetch_list=[output])
                np.testing.assert_allclose(np_result, result, rtol=1e-05)

class TestMseInvalidInput(unittest.TestCase):

    @test_with_pir_api
    def test_error(self):
        if False:
            while True:
                i = 10

        def test_invalid_input():
            if False:
                print('Hello World!')
            input = [256, 3]
            label = paddle.static.data(name='label1', shape=[None, 3], dtype='float32')
            loss = paddle.nn.functional.mse_loss(input, label)
        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_label():
            if False:
                while True:
                    i = 10
            input = paddle.static.data(name='input1', shape=[None, 3], dtype='float32')
            label = [256, 3]
            loss = paddle.nn.functional.mse_loss(input, label)
        self.assertRaises(TypeError, test_invalid_label)

class TestNNMseLoss(unittest.TestCase):

    @test_with_pir_api
    def test_NNMseLoss_mean(self):
        if False:
            for i in range(10):
                print('nop')
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            label_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            paddle.enable_static()
            prog = base.Program()
            startup_prog = base.Program()
            place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
            with base.program_guard(prog, startup_prog):
                input = paddle.static.data(name='input', shape=dim, dtype='float32')
                label = paddle.static.data(name='label', shape=dim, dtype='float32')
                mse_loss = paddle.nn.loss.MSELoss()
                ret = mse_loss(input, label)
                exe = base.Executor(place)
                (static_result,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            with base.dygraph.guard():
                mse_loss = paddle.nn.loss.MSELoss()
                dy_ret = mse_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
                dy_result = dy_ret.numpy()
            sub = input_np - label_np
            expected = np.mean(sub * sub)
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            self.assertEqual(dy_result.shape, ())

    @test_with_pir_api
    def test_NNMseLoss_sum(self):
        if False:
            print('Hello World!')
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            label_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            paddle.enable_static()
            prog = base.Program()
            startup_prog = base.Program()
            place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
            with base.program_guard(prog, startup_prog):
                input = paddle.static.data(name='input', shape=dim, dtype='float32')
                label = paddle.static.data(name='label', shape=dim, dtype='float32')
                mse_loss = paddle.nn.loss.MSELoss(reduction='sum')
                ret = mse_loss(input, label)
                exe = base.Executor(place)
                (static_result,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            with base.dygraph.guard():
                mse_loss = paddle.nn.loss.MSELoss(reduction='sum')
                dy_ret = mse_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
                dy_result = dy_ret.numpy()
            sub = input_np - label_np
            expected = np.sum(sub * sub)
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            self.assertEqual(dy_result.shape, ())

    @test_with_pir_api
    def test_NNMseLoss_none(self):
        if False:
            while True:
                i = 10
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            label_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            paddle.enable_static()
            prog = base.Program()
            startup_prog = base.Program()
            place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
            with base.program_guard(prog, startup_prog):
                input = paddle.static.data(name='input', shape=dim, dtype='float32')
                label = paddle.static.data(name='label', shape=dim, dtype='float32')
                mse_loss = paddle.nn.loss.MSELoss(reduction='none')
                ret = mse_loss(input, label)
                exe = base.Executor(place)
                (static_result,) = exe.run(prog, feed={'input': input_np, 'label': label_np}, fetch_list=[ret])
            with base.dygraph.guard():
                mse_loss = paddle.nn.loss.MSELoss(reduction='none')
                dy_ret = mse_loss(base.dygraph.to_variable(input_np), base.dygraph.to_variable(label_np))
                dy_result = dy_ret.numpy()
            sub = input_np - label_np
            expected = sub * sub
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            self.assertEqual(dy_result.shape, tuple(dim))

class TestNNFunctionalMseLoss(unittest.TestCase):

    @test_with_pir_api
    def test_NNFunctionalMseLoss_mean(self):
        if False:
            return 10
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            target_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            paddle.enable_static()
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(name='input', shape=dim, dtype='float32')
                target = paddle.static.data(name='target', shape=dim, dtype='float32')
                mse_loss = paddle.nn.functional.mse_loss(input, target, 'mean')
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            (static_result,) = exe.run(prog, feed={'input': input_np, 'target': target_np}, fetch_list=[mse_loss])
            paddle.disable_static()
            dy_ret = paddle.nn.functional.mse_loss(paddle.to_tensor(input_np), paddle.to_tensor(target_np), 'mean')
            dy_result = dy_ret.numpy()
            sub = input_np - target_np
            expected = np.mean(sub * sub)
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            self.assertEqual(dy_result.shape, ())

    @test_with_pir_api
    def test_NNFunctionalMseLoss_sum(self):
        if False:
            i = 10
            return i + 15
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            target_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            paddle.enable_static()
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(name='input', shape=dim, dtype='float32')
                target = paddle.static.data(name='target', shape=dim, dtype='float32')
                mse_loss = paddle.nn.functional.mse_loss(input, target, 'sum')
                exe = paddle.static.Executor(place)
                exe.run(startup_prog)
                (static_result,) = exe.run(prog, feed={'input': input_np, 'target': target_np}, fetch_list=[mse_loss])
            paddle.disable_static()
            dy_ret = paddle.nn.functional.mse_loss(paddle.to_tensor(input_np), paddle.to_tensor(target_np), 'sum')
            dy_result = dy_ret.numpy()
            sub = input_np - target_np
            expected = np.sum(sub * sub)
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            self.assertEqual(dy_result.shape, ())

    @test_with_pir_api
    def test_NNFunctionalMseLoss_none(self):
        if False:
            while True:
                i = 10
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            target_np = np.random.uniform(0.1, 0.5, dim).astype('float32')
            paddle.enable_static()
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(name='input', shape=dim, dtype='float32')
                target = paddle.static.data(name='target', shape=dim, dtype='float32')
                mse_loss = paddle.nn.functional.mse_loss(input, target, 'none')
                exe = paddle.static.Executor(place)
                exe.run(startup_prog)
                (static_result,) = exe.run(prog, feed={'input': input_np, 'target': target_np}, fetch_list=[mse_loss])
            paddle.disable_static()
            dy_ret = paddle.nn.functional.mse_loss(paddle.to_tensor(input_np), paddle.to_tensor(target_np), 'none')
            dy_result = dy_ret.numpy()
            sub = input_np - target_np
            expected = sub * sub
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            self.assertEqual(dy_result.shape, tuple(dim))
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()