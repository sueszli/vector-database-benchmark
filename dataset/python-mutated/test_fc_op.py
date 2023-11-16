import unittest
import numpy as np
from op_test import OpTest, paddle_static_guard
import paddle
from paddle import base
from paddle.base import Program, core, program_guard
SEED = 2020

def fc_refer(matrix, with_bias, with_relu=False):
    if False:
        return 10
    (in_n, in_c, in_h, in_w) = matrix.input.shape
    (w_i, w_o) = matrix.weights.shape
    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    w_data = np.reshape(matrix.weights, [w_i, w_o])
    b_data = np.reshape(matrix.bias, [1, w_o])
    result = None
    if with_bias:
        result = np.dot(x_data, w_data) + b_data
    else:
        result = np.dot(x_data, w_data)
    if with_relu:
        return np.maximum(result, 0)
    else:
        return result

class MatrixGenerate:

    def __init__(self, mb, ic, oc, h, w, bias_dims=2):
        if False:
            print('Hello World!')
        self.input = np.random.random((mb, ic, h, w)).astype('float32')
        self.weights = np.random.random((ic * h * w, oc)).astype('float32')
        if bias_dims == 2:
            self.bias = np.random.random((1, oc)).astype('float32')
        else:
            self.bias = np.random.random(oc).astype('float32')

class TestFCOp(OpTest):

    def config(self):
        if False:
            while True:
                i = 10
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3, 2)

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fc'
        self.config()
        if self.with_bias:
            self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights, 'Bias': self.matrix.bias}
        else:
            self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights}
        if self.with_relu:
            activation_type = 'relu'
        else:
            activation_type = ''
        self.attrs = {'use_mkldnn': False, 'activation_type': activation_type}
        self.outputs = {'Out': fc_refer(self.matrix, self.with_bias, self.with_relu)}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_dygraph=False)

class TestFCOpNoBias1(TestFCOp):

    def config(self):
        if False:
            print('Hello World!')
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(2, 8, 10, 1, 1, 2)

class TestFCOpNoBias2(TestFCOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)

class TestFCOpNoBias4(TestFCOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(1, 32, 64, 3, 3, 1)

class TestFCOpWithBias1(TestFCOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.with_bias = True
        self.with_relu = False
        self.matrix = MatrixGenerate(3, 8, 10, 2, 1, 2)

class TestFCOpWithBias2(TestFCOp):

    def config(self):
        if False:
            print('Hello World!')
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)

class TestFCOpWithBias3(TestFCOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 64, 32, 3, 3, 1)

class TestFCOpWithPadding(TestFCOp):

    def config(self):
        if False:
            print('Hello World!')
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 4, 3, 128, 128, 2)

class TestFcOp_NumFlattenDims_NegOne(unittest.TestCase):

    def test_api(self):
        if False:
            while True:
                i = 10

        def run_program(num_flatten_dims):
            if False:
                return 10
            paddle.seed(SEED)
            np.random.seed(SEED)
            startup_program = Program()
            main_program = Program()
            with paddle_static_guard():
                with program_guard(main_program, startup_program):
                    input = np.random.random([2, 2, 25]).astype('float32')
                    x = paddle.static.data(name='x', shape=[2, 2, 25], dtype='float32')
                    out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=num_flatten_dims)
                place = base.CPUPlace() if not core.is_compiled_with_cuda() else base.CUDAPlace(0)
                exe = base.Executor(place=place)
                exe.run(startup_program)
                out = exe.run(main_program, feed={'x': input}, fetch_list=[out])
                return out
        res_1 = run_program(-1)
        res_2 = run_program(2)
        np.testing.assert_array_equal(res_1, res_2)

class TestFCOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype('float32')

            def test_Variable():
                if False:
                    print('Hello World!')
                with paddle_static_guard():
                    paddle.static.nn.fc(x=input_data, size=1)
            self.assertRaises(TypeError, test_Variable)

            def test_input_list():
                if False:
                    return 10
                with paddle_static_guard():
                    paddle.static.nn.fc(x=[input_data], size=1)
            self.assertRaises(TypeError, test_input_list)

            def test_type():
                if False:
                    for i in range(10):
                        print('nop')
                with paddle_static_guard():
                    x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype='int32')
                    paddle.static.nn.fc(x=x2, size=1)
            self.assertRaises(TypeError, test_type)
            with paddle_static_guard():
                x3 = paddle.static.data(name='x3', shape=[-1, 4], dtype='float16')
                paddle.static.nn.fc(x=x3, size=1)
if __name__ == '__main__':
    unittest.main()