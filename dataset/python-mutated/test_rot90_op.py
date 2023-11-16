import unittest
import numpy as np
import paddle
from paddle import base

class TestRot90_API(unittest.TestCase):
    """Test rot90 api."""

    def test_static_graph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, axes=[0, 1])
            output = paddle.rot90(output, k=1, axes=[0, 1])
            output = output.rot90(k=1, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_k_0(self):
        if False:
            return 10
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=0, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_k_2(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=2, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[6, 5, 4], [3, 2, 1]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_k_3(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=3, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_neg_k_1(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=-1, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_neg_k_2(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=-2, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[6, 5, 4], [3, 2, 1]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_neg_k_3(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=-3, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[3, 6], [2, 5], [1, 4]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_static_neg_k_4(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=-4, axes=[0, 1])
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program, feed={'input': img}, fetch_list=[output])
            out_np = np.array(res[0])
            out_ref = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            self.assertTrue((out_np == out_ref).all(), msg='rot90 output is wrong, out =' + str(out_np))

    def test_error_api(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()

        def run1():
            if False:
                print('Hello World!')
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, axes=[0])
        self.assertRaises(ValueError, run1)

        def run2():
            if False:
                print('Hello World!')
            input = paddle.static.data(name='input', dtype='float32', shape=[2])
            output = paddle.rot90(input, k=1, axes=[0, 1])
        self.assertRaises(ValueError, run2)

        def run3():
            if False:
                return 10
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, axes=[0, 0])
        self.assertRaises(ValueError, run3)

        def run4():
            if False:
                while True:
                    i = 10
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, axes=[3, 1])
        self.assertRaises(ValueError, run4)

        def run5():
            if False:
                while True:
                    i = 10
            input = paddle.static.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, axes=[0, 3])
        self.assertRaises(ValueError, run5)

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        with base.dygraph.guard():
            inputs = base.dygraph.to_variable(img)
            ret = paddle.rot90(inputs, k=1, axes=[0, 1])
            ret = ret.rot90(1, axes=[0, 1])
            ret = paddle.rot90(ret, k=1, axes=[0, 1])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)
            self.assertTrue((ret.numpy() == out_ref).all(), msg='rot90 output is wrong, out =' + str(ret.numpy()))
if __name__ == '__main__':
    unittest.main()