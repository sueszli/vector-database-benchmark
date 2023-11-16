import unittest
import numpy as np
import paddle
from paddle import base, rand
from paddle.base import Program, core, program_guard

class TestRandOpError(unittest.TestCase):
    """
    This class test the input type check.
    """

    def test_errors(self):
        if False:
            print('Hello World!')
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                if False:
                    i = 10
                    return i + 15
                x1 = base.create_lod_tensor(np.zeros((4, 784)), [[1, 1, 1, 1]], base.CPUPlace())
                rand(x1)
            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                if False:
                    return 10
                dim_1 = paddle.tensor.fill_constant([1], 'int64', 3)
                dim_2 = paddle.tensor.fill_constant([1], 'int32', 5)
                rand(shape=[dim_1, dim_2], dtype='int32')
            self.assertRaises(TypeError, test_dtype)

class TestRandOp(unittest.TestCase):
    """
    This class test the common usages of randop.
    """

    def run_net(self, use_cuda=False):
        if False:
            for i in range(10):
                print('nop')
        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        exe = base.Executor(place)
        train_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(train_program, startup_program):
            result_0 = rand([3, 4])
            result_1 = rand([3, 4], 'float64')
            dim_1 = paddle.tensor.fill_constant([1], 'int64', 3)
            dim_2 = paddle.tensor.fill_constant([1], 'int32', 5)
            result_2 = rand(shape=[dim_1, dim_2])
            var_shape = paddle.static.data(name='var_shape', shape=[2], dtype='int64')
            result_3 = rand(var_shape)
            var_shape_int32 = paddle.static.data(name='var_shape_int32', shape=[2], dtype='int32')
            result_4 = rand(var_shape_int32)
        exe.run(startup_program)
        x1 = np.array([3, 2]).astype('int64')
        x2 = np.array([4, 3]).astype('int32')
        ret = exe.run(train_program, feed={'var_shape': x1, 'var_shape_int32': x2}, fetch_list=[result_1, result_1, result_2, result_3, result_4])

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_net(False)
        if core.is_compiled_with_cuda():
            self.run_net(True)

class TestRandOpForDygraph(unittest.TestCase):
    """
    This class test the common usages of randop.
    """

    def run_net(self, use_cuda=False):
        if False:
            i = 10
            return i + 15
        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        with base.dygraph.guard(place):
            rand([3, 4])
            rand([3, 4], 'float64')
            dim_1 = paddle.tensor.fill_constant([1], 'int64', 3)
            dim_2 = paddle.tensor.fill_constant([1], 'int32', 5)
            rand(shape=[dim_1, dim_2])
            var_shape = paddle.to_tensor(np.array([3, 4]))
            rand(var_shape)

    def test_run(self):
        if False:
            print('Hello World!')
        self.run_net(False)
        if core.is_compiled_with_cuda():
            self.run_net(True)

class TestRandDtype(unittest.TestCase):

    def test_default_dtype(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()

        def test_default_fp16():
            if False:
                while True:
                    i = 10
            paddle.framework.set_default_dtype('float16')
            out = paddle.tensor.random.rand([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP16)

        def test_default_fp32():
            if False:
                print('Hello World!')
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.rand([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP32)

        def test_default_fp64():
            if False:
                return 10
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.rand([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP64)
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            test_default_fp16()
        test_default_fp64()
        test_default_fp32()
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()