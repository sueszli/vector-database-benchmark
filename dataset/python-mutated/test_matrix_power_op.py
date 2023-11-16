import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base, static
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

class TestMatrixPowerOp(OpTest):

    def config(self):
        if False:
            while True:
                i = 10
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 0

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'matrix_power'
        self.python_api = paddle.tensor.matrix_power
        self.config()
        np.random.seed(123)
        mat = np.random.random(self.matrix_shape).astype(self.dtype)
        powered_mat = np.linalg.matrix_power(mat, self.n)
        self.inputs = {'X': mat}
        self.outputs = {'Out': powered_mat}
        self.attrs = {'n': self.n}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', numeric_grad_delta=1e-05, max_relative_error=1e-07, check_pir=True)

class TestMatrixPowerOpN1(TestMatrixPowerOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 1

class TestMatrixPowerOpN2(TestMatrixPowerOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 2

class TestMatrixPowerOpN3(TestMatrixPowerOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 3

class TestMatrixPowerOpN4(TestMatrixPowerOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 4

class TestMatrixPowerOpN5(TestMatrixPowerOp):

    def config(self):
        if False:
            print('Hello World!')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 5

class TestMatrixPowerOpN6(TestMatrixPowerOp):

    def config(self):
        if False:
            return 10
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 6

class TestMatrixPowerOpN10(TestMatrixPowerOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 10

class TestMatrixPowerOpNMinus(TestMatrixPowerOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -1

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', numeric_grad_delta=1e-05, max_relative_error=1e-06, check_pir=True)

class TestMatrixPowerOpNMinus2(TestMatrixPowerOpNMinus):

    def config(self):
        if False:
            return 10
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -2

class TestMatrixPowerOpNMinus3(TestMatrixPowerOpNMinus):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -3

class TestMatrixPowerOpNMinus4(TestMatrixPowerOpNMinus):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -4

class TestMatrixPowerOpNMinus5(TestMatrixPowerOpNMinus):

    def config(self):
        if False:
            print('Hello World!')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -5

class TestMatrixPowerOpNMinus6(TestMatrixPowerOpNMinus):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -6

class TestMatrixPowerOpNMinus10(TestMatrixPowerOp):

    def config(self):
        if False:
            print('Hello World!')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = -10

    def test_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', numeric_grad_delta=1e-05, max_relative_error=1e-06, check_pir=True)

class TestMatrixPowerOpBatched1(TestMatrixPowerOp):

    def config(self):
        if False:
            print('Hello World!')
        self.matrix_shape = [8, 4, 4]
        self.dtype = 'float64'
        self.n = 5

class TestMatrixPowerOpBatched2(TestMatrixPowerOp):

    def config(self):
        if False:
            print('Hello World!')
        self.matrix_shape = [2, 6, 4, 4]
        self.dtype = 'float64'
        self.n = 4

class TestMatrixPowerOpBatched3(TestMatrixPowerOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [2, 6, 4, 4]
        self.dtype = 'float64'
        self.n = 0

class TestMatrixPowerOpBatchedLong(TestMatrixPowerOp):

    def config(self):
        if False:
            return 10
        self.matrix_shape = [1, 2, 3, 4, 4, 3, 3]
        self.dtype = 'float64'
        self.n = 3

class TestMatrixPowerOpLarge1(TestMatrixPowerOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.matrix_shape = [32, 32]
        self.dtype = 'float64'
        self.n = 3

class TestMatrixPowerOpLarge2(TestMatrixPowerOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.n = 32

class TestMatrixPowerOpFP32(TestMatrixPowerOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float32'
        self.n = 2

    def test_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', max_relative_error=0.01, check_pir=True)

class TestMatrixPowerOpBatchedFP32(TestMatrixPowerOpFP32):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [2, 8, 4, 4]
        self.dtype = 'float32'
        self.n = 2

class TestMatrixPowerOpLarge1FP32(TestMatrixPowerOpFP32):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.matrix_shape = [32, 32]
        self.dtype = 'float32'
        self.n = 2

class TestMatrixPowerOpLarge2FP32(TestMatrixPowerOpFP32):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float32'
        self.n = 32

class TestMatrixPowerOpFP32Minus(TestMatrixPowerOpFP32):

    def config(self):
        if False:
            while True:
                i = 10
        self.matrix_shape = [10, 10]
        self.dtype = 'float32'
        self.n = -1

class TestMatrixPowerAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            for i in range(10):
                print('nop')
        with static.program_guard(static.Program(), static.Program()):
            input_x = paddle.static.data(name='input_x', shape=[4, 4], dtype='float64')
            result = paddle.linalg.matrix_power(x=input_x, n=-2)
            input_np = np.random.random([4, 4]).astype('float64')
            result_np = np.linalg.matrix_power(input_np, -2)
            exe = base.Executor(place)
            fetches = exe.run(feed={'input_x': input_np}, fetch_list=[result])
            np.testing.assert_allclose(fetches[0], np.linalg.matrix_power(input_np, -2), rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        if False:
            print('Hello World!')
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([4, 4]).astype('float64')
                input = paddle.to_tensor(input_np)
                result = paddle.linalg.matrix_power(input, -2)
                np.testing.assert_allclose(result.numpy(), np.linalg.matrix_power(input_np, -2), rtol=1e-05)

class TestMatrixPowerAPIError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        input_np = np.random.random([4, 4]).astype('float64')
        self.assertRaises(TypeError, paddle.linalg.matrix_power, input_np)
        for n in [2.0, '2', -2.0]:
            input = paddle.static.data(name='input_float32', shape=[4, 4], dtype='float32')
            self.assertRaises(TypeError, paddle.linalg.matrix_power, input, n)
        for dtype in ['bool', 'int32', 'int64', 'float16']:
            input = paddle.static.data(name='input_' + dtype, shape=[4, 4], dtype=dtype)
            self.assertRaises(TypeError, paddle.linalg.matrix_power, input, 2)
        input = paddle.static.data(name='input_1', shape=[4, 4], dtype='float32')
        out = paddle.static.data(name='output', shape=[4, 4], dtype='float64')
        self.assertRaises(TypeError, paddle.linalg.matrix_power, input, 2, out)
        input = paddle.static.data(name='input_2', shape=[4], dtype='float32')
        self.assertRaises(ValueError, paddle.linalg.matrix_power, input, 2)
        input = paddle.static.data(name='input_3', shape=[4, 5], dtype='float32')
        self.assertRaises(ValueError, paddle.linalg.matrix_power, input, 2)
        input = paddle.static.data(name='input_4', shape=[1, 1, 0, 0], dtype='float32')
        self.assertRaises(ValueError, paddle.linalg.matrix_power, input, 2)
        input = paddle.static.data(name='input_5', shape=[0, 0], dtype='float32')
        self.assertRaises(ValueError, paddle.linalg.matrix_power, input, -956301312)

class TestMatrixPowerSingularAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            return 10
        with static.program_guard(static.Program(), static.Program()):
            input = paddle.static.data(name='input', shape=[4, 4], dtype='float64')
            result = paddle.linalg.matrix_power(x=input, n=-2)
            input_np = np.zeros([4, 4]).astype('float64')
            exe = base.Executor(place)
            try:
                fetches = exe.run(feed={'input': input_np}, fetch_list=[result])
            except RuntimeError as ex:
                print('The mat is singular')
            except ValueError as ex:
                print('The mat is singular')

    @test_with_pir_api
    def test_static(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        for place in self.places:
            self.check_static_result(place=place)
        paddle.disable_static()

    def test_dygraph(self):
        if False:
            print('Hello World!')
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.ones([4, 4]).astype('float64')
                input = base.dygraph.to_variable(input_np)
                try:
                    result = paddle.linalg.matrix_power(input, -2)
                except RuntimeError as ex:
                    print('The mat is singular')
                except ValueError as ex:
                    print('The mat is singular')
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()