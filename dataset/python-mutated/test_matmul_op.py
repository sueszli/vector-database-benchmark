import unittest
import numpy as np
from op_test import OpTest, paddle_static_guard
import paddle
from paddle import base

def generate_compatible_shapes(dim_X, dim_Y, transpose_X, transpose_Y):
    if False:
        for i in range(10):
            print('nop')
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 5
    if dim_X == 1 and transpose_X or (dim_Y == 1 and transpose_Y):
        K = 1
    if dim_X == 1:
        if transpose_X:
            shape_X = [M]
        else:
            shape_X = [K]
    if dim_Y == 1:
        if transpose_Y:
            shape_Y = [N]
        else:
            shape_Y = [K]
    if dim_X >= 2:
        if transpose_X:
            shape_X = [K, M]
        else:
            shape_X = [M, K]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [N, K]
        else:
            shape_Y = [K, N]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return (shape_X, shape_Y)

def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    if False:
        while True:
            i = 10
    'Reference forward implementation using np.matmul.'
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size, 1))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = list(range(len(X.shape)))
            (dim[-1], dim[len(X.shape) - 2]) = (dim[len(X.shape) - 2], dim[-1])
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((1, Y.size))
        else:
            dim = list(range(len(Y.shape)))
            (dim[-1], dim[len(Y.shape) - 2]) = (dim[len(Y.shape) - 2], dim[-1])
            Y = np.transpose(Y, tuple(dim))
    Out = np.matmul(X, Y)
    return Out

class Generator:

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'matmul'
        X = np.random.random(self.shape_X).astype('float32')
        Y = np.random.random(self.shape_Y).astype('float32')
        Out = reference_matmul(X, Y, self.transpose_X, self.transpose_Y)
        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {'transpose_X': self.transpose_X, 'transpose_Y': self.transpose_Y}
        self.outputs = {'Out': Out}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_cinn=True)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.001, check_cinn=True)

    def test_check_grad_ignore_x(self):
        if False:
            print('Hello World!')
        self.check_grad(['Y'], 'Out', max_relative_error=0.001, no_grad_set=set('X'), check_cinn=True)

    def test_check_grad_ignore_y(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', max_relative_error=0.001, no_grad_set=set('Y'), check_cinn=True)

def generate_compatible_shapes_ndim(dim, transpose_X, transpose_Y):
    if False:
        print('Hello World!')
    M = 2
    N = 4
    K = 3
    shape_X = [2 for _ in range(dim - 2)]
    shape_Y = [2 for _ in range(dim - 2)]
    if transpose_X:
        shape_X += [K, M]
    else:
        shape_X += [M, K]
    if transpose_Y:
        shape_Y += [N, K]
    else:
        shape_Y += [K, N]
    return (shape_X, shape_Y)
for dim in [4]:
    for transpose_X in [False, True]:
        for transpose_Y in [False, True]:
            test_name = 'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(dim, dim, transpose_X, transpose_Y)
            (shape_X, shape_Y) = generate_compatible_shapes_ndim(dim, transpose_X, transpose_Y)
            globals()[test_name] = type(test_name, (Generator, OpTest), {'shape_X': shape_X, 'shape_Y': shape_Y, 'transpose_X': transpose_X, 'transpose_Y': transpose_Y})

class API_TestMm(unittest.TestCase):

    def test_out(self):
        if False:
            while True:
                i = 10
        with paddle_static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data(name='x', shape=[2], dtype='float64')
                y = paddle.static.data(name='y', shape=[2], dtype='float64')
                result = paddle.mm(x, y)
                exe = base.Executor(base.CPUPlace())
                data1 = np.random.rand(2)
                data2 = np.random.rand(2)
                np_res = exe.run(feed={'x': data1, 'y': data2}, fetch_list=[result])
                expected_result = np.matmul(data1, data2)
            np.testing.assert_allclose(np_res, expected_result, rtol=1e-05, atol=1e-05, err_msg='two value is            {}\n{}, check diff!'.format(np_res, expected_result))

    def test_dygraph_without_out(self):
        if False:
            return 10
        device = base.CPUPlace()
        with base.dygraph.guard(device):
            input_array1 = np.random.rand(3, 4).astype('float64')
            input_array2 = np.random.rand(4, 3).astype('float64')
            data1 = base.dygraph.to_variable(input_array1)
            data2 = base.dygraph.to_variable(input_array2)
            out = paddle.mm(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
        np.testing.assert_allclose(expected_result, out.numpy(), rtol=1e-05)

class Test_API_Matmul(unittest.TestCase):

    def test_dygraph_without_out(self):
        if False:
            for i in range(10):
                print('nop')
        device = base.CPUPlace()
        with base.dygraph.guard(device):
            input_array1 = np.random.rand(3, 4).astype('float64')
            input_array2 = np.random.rand(4, 3).astype('float64')
            data1 = base.dygraph.to_variable(input_array1)
            data2 = base.dygraph.to_variable(input_array2)
            out = paddle.matmul(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
        np.testing.assert_allclose(expected_result, out.numpy(), rtol=1e-05)

class API_TestMmError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with paddle_static_guard():

            def test_error1():
                if False:
                    return 10
                with base.program_guard(base.Program(), base.Program()):
                    data1 = paddle.static.data(name='data1', shape=[10, 2], dtype='float32')
                    data2 = paddle.static.data(name='data2', shape=[3, 10], dtype='float32')
                    paddle.mm(data1, data2)
            self.assertRaises(ValueError, test_error1)

            def test_error2():
                if False:
                    i = 10
                    return i + 15
                with base.program_guard(base.Program(), base.Program()):
                    data1 = paddle.static.data(name='data1', shape=[-1, 10, 2], dtype='float32')
                    data2 = paddle.static.data(name='data2', shape=[-1, 2, 10], dtype='float32')
                    paddle.mm(data1, data2)
            test_error2()

            def test_error3():
                if False:
                    for i in range(10):
                        print('nop')
                with base.program_guard(base.Program(), base.Program()):
                    data1 = paddle.static.data(name='data1', shape=[10, 10, 2], dtype='float32')
                    data2 = paddle.static.data(name='data2', shape=[3, 2, 10], dtype='float32')
                    paddle.mm(data1, data2)
            self.assertRaises(ValueError, test_error3)
if __name__ == '__main__':
    unittest.main()