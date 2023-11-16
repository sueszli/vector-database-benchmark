import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.static import Program, program_guard

class TestRandnOp(unittest.TestCase):

    def test_api(self):
        if False:
            return 10
        shape = [1000, 784]
        train_program = Program()
        startup_program = Program()
        with program_guard(train_program, startup_program):
            x1 = paddle.randn(shape, 'float32')
            x2 = paddle.randn(shape, 'float64')
            dim_1 = paddle.tensor.fill_constant([1], 'int64', 20)
            dim_2 = paddle.tensor.fill_constant([1], 'int32', 50)
            x3 = paddle.randn([dim_1, dim_2, 784])
            var_shape = paddle.static.data('X', [2], 'int32')
            x4 = paddle.randn(var_shape)
        place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        res = exe.run(train_program, feed={'X': np.array(shape, dtype='int32')}, fetch_list=[x1, x2, x3, x4])
        for out in res:
            self.assertAlmostEqual(np.mean(out), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(out), 1.0, delta=0.1)

class TestRandnOpForDygraph(unittest.TestCase):

    def test_api(self):
        if False:
            i = 10
            return i + 15
        shape = [1000, 784]
        place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        paddle.disable_static(place)
        x1 = paddle.randn(shape, 'float32')
        x2 = paddle.randn(shape, 'float64')
        dim_1 = paddle.tensor.fill_constant([1], 'int64', 20)
        dim_2 = paddle.tensor.fill_constant([1], 'int32', 50)
        x3 = paddle.randn(shape=[dim_1, dim_2, 784])
        var_shape = paddle.to_tensor(np.array(shape))
        x4 = paddle.randn(var_shape)
        for out in [x1, x2, x3, x4]:
            self.assertAlmostEqual(np.mean(out.numpy()), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(out.numpy()), 1.0, delta=0.1)
        paddle.enable_static()

class TestRandnOpError(unittest.TestCase):

    def test_error(self):
        if False:
            for i in range(10):
                print('nop')
        with program_guard(Program(), Program()):
            self.assertRaises(TypeError, paddle.randn, 1)
            self.assertRaises(TypeError, paddle.randn, [1, 2], 'int32')
if __name__ == '__main__':
    unittest.main()