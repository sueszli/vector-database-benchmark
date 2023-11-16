import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle import base

def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    if False:
        print('Hello World!')
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
        elif Y.ndim == 2:
            Y = Y.T
        else:
            dim = list(range(len(Y.shape)))
            (dim[-1], dim[len(Y.shape) - 2]) = (dim[len(Y.shape) - 2], dim[-1])
            Y = np.transpose(Y, tuple(dim))
    if X.ndim == 3 and Y.ndim == 2:
        x_dims = X.shape
        X = X.reshape((x_dims[0] * x_dims[1], x_dims[2]))
    if Y.ndim == 3 and X.ndim == 2:
        y_dims = Y.shape
        Y = Y.reshape((y_dims[0] * y_dims[1], y_dims[2]))
    Out = np.matmul(X, Y)
    return Out

def generate_compatible_shapes(dim_X, dim_Y, transpose_X, transpose_Y, batch_size):
    if False:
        print('Hello World!')
    BATCH_SIZE = 2
    if batch_size is not None:
        BATCH_SIZE = batch_size
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
    if dim_Y == 3 and dim_X == 2:
        if not transpose_X:
            shape_X[1] = shape_X[1] * BATCH_SIZE
        else:
            shape_X[0] = shape_X[0] * BATCH_SIZE
    return (shape_X, shape_Y)

def generate_compatible_shapes_2(dim, transpose_X, transpose_Y):
    if False:
        for i in range(10):
            print('nop')
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

class XPUTestMatmulOpErr(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'matmul'
        self.use_dynamic_create_class = False

    class API_TestMm(unittest.TestCase):

        def test_out(self):
            if False:
                print('Hello World!')
            with base.program_guard(base.Program()):
                x = paddle.static.data(name='x', shape=[2], dtype=self.in_type)
                y = paddle.static.data(name='y', shape=[2], dtype=self.in_type)
                result = paddle.mm(x, y)
                exe = base.Executor(base.XPUPlace(0))
                data1 = np.random.rand(2).astype(self.in_type)
                data2 = np.random.rand(2).astype(self.in_type)
                np_res = exe.run(feed={'x': data1, 'y': data2}, fetch_list=[result])
                expected_result = np.matmul(data1, data2)
                np.testing.assert_allclose(np_res, expected_result, atol=0.001)

        def test_dygraph_without_out(self):
            if False:
                return 10
            device = base.XPUPlace(0)
            with base.dygraph.guard(device):
                input_array1 = np.random.rand(3, 4).astype(self.in_type)
                input_array2 = np.random.rand(4, 3).astype(self.in_type)
                data1 = base.dygraph.to_variable(input_array1)
                data2 = base.dygraph.to_variable(input_array2)
                out = paddle.mm(data1, data2)
                expected_result = np.matmul(input_array1, input_array2)
                np.testing.assert_allclose(expected_result, out.numpy(), atol=0.001)

    class Test_API_Matmul(unittest.TestCase):

        def test_dygraph_without_out(self):
            if False:
                for i in range(10):
                    print('nop')
            device = base.XPUPlace(0)
            with base.dygraph.guard(device):
                input_array1 = np.random.rand(3, 4).astype(self.in_type)
                input_array2 = np.random.rand(4, 3).astype(self.in_type)
                data1 = base.dygraph.to_variable(input_array1).astype(self.in_type)
                data2 = base.dygraph.to_variable(input_array2).astype(self.in_type)
                out = paddle.matmul(data1, data2)
                expected_result = np.matmul(input_array1, input_array2)
                np.testing.assert_allclose(expected_result, out.numpy(), atol=0.001)

    class API_TestMmError(unittest.TestCase):

        def test_errors(self):
            if False:
                for i in range(10):
                    print('nop')

            def test_error1():
                if False:
                    i = 10
                    return i + 15
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
                    while True:
                        i = 10
                with base.program_guard(base.Program(), base.Program()):
                    data1 = paddle.static.data(name='data1', shape=[10, 10, 2], dtype='float32')
                    data2 = paddle.static.data(name='data2', shape=[3, 2, 10], dtype='float32')
                    paddle.mm(data1, data2)
            self.assertRaises(ValueError, test_error3)

class TestMatmulBaseGenerator(XPUOpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'matmul'
        self.dtype = np.float32 if not hasattr(self, 'in_type') else self.in_type
        self.__class__.no_need_check_grad = False if not hasattr(self, 'no_need_check_grad') else self.no_need_check_grad
        shape_X = [4, 5] if not hasattr(self, 'shape_X') else self.shape_X
        shape_Y = [5, 6] if not hasattr(self, 'shape_Y') else self.shape_Y
        transpose_X = False if not hasattr(self, 'transpose_X') else self.transpose_X
        transpose_Y = False if not hasattr(self, 'transpose_Y') else self.transpose_Y
        X = np.random.random(shape_X).astype(self.dtype)
        Y = np.random.random(shape_Y).astype(self.dtype)
        Out = reference_matmul(X, Y, transpose_X, transpose_Y).astype(self.dtype)
        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {'transpose_X': transpose_X, 'transpose_Y': transpose_Y}
        self.outputs = {'Out': Out}

    def test_check_output(self):
        if False:
            return 10
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, atol=0.001)

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self.__class__, 'no_need_check_grad') and self.__class__.no_need_check_grad:
            return
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, ['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ignore_x(self):
        if False:
            return 10
        if hasattr(self.__class__, 'no_need_check_grad') and self.__class__.no_need_check_grad:
            return
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, ['Y'], 'Out', max_relative_error=0.05, no_grad_set=set('X'))

    def test_check_grad_ignore_y(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self.__class__, 'no_need_check_grad') and self.__class__.no_need_check_grad:
            return
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Y'))

class XPUTestMatmulOp1(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'matmul'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        if False:
            return 10
        base_class = TestMatmulBaseGenerator
        classes = []
        xpu_support_dims_list = [[1, 1], [2, 2], [3, 3]]
        batch_size = [2, 4, 5, 10, 50, 100, 300]
        for dims in xpu_support_dims_list:
            dim_X = dims[0]
            dim_Y = dims[1]
            for transose_x in [True, False]:
                for transose_y in [True, False]:
                    for batch in batch_size:
                        no_need_check_grad = False
                        if batch >= 5:
                            no_need_check_grad = True
                        class_name = 'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}_batch_{}'.format(dim_X, dim_Y, transose_x, transose_y, batch)
                        (shape_x, shape_y) = generate_compatible_shapes(dim_X, dim_Y, transose_x, transose_y, batch)
                        attr_dict = {'shape_X': shape_x, 'shape_Y': shape_y, 'transpose_X': transose_x, 'transpose_Y': transose_y, 'no_need_check_grad': no_need_check_grad, 'op_type': 'matmul'}
                        classes.append([class_name, attr_dict])
        return (base_class, classes)

class XPUTestMatmulOp3(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'matmul'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        if False:
            while True:
                i = 10
        base_class = TestMatmulBaseGenerator
        classes = []
        for dim in [4]:
            for transpose_X in [False, True]:
                for transpose_Y in [False, True]:
                    class_name = 'TestMatMulOp2_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(dim, dim, transpose_X, transpose_Y)
                    (shape_X, shape_Y) = generate_compatible_shapes_2(dim, transpose_X, transpose_Y)
                    attr_dict = {'shape_X': shape_X, 'shape_Y': shape_Y, 'transpose_X': transpose_X, 'transpose_Y': transpose_Y, 'op_type': 'matmul'}
                    classes.append([class_name, attr_dict])
        return (base_class, classes)
support_types = get_xpu_op_support_types('matmul')
for stype in support_types:
    create_test_class(globals(), XPUTestMatmulOpErr, stype)
    create_test_class(globals(), XPUTestMatmulOp1, stype)
    create_test_class(globals(), XPUTestMatmulOp3, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()