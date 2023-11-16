import sys
import unittest
import numpy as np
import paddle
from paddle.base import core
sys.path.append('..')
from op_test import OpTest
from paddle import base
from paddle.base import Program, program_guard

class TestSolveOp(OpTest):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.python_api = paddle.linalg.solve
        self.input_x_matrix_shape = [15, 15]
        self.input_y_matrix_shape = [15, 10]
        self.dtype = 'float64'

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        self.config()
        self.op_type = 'solve'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random(self.input_x_matrix_shape).astype(self.dtype), 'Y': np.random.random(self.input_y_matrix_shape).astype(self.dtype)}
        self.outputs = {'Out': np.linalg.solve(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

class TestSolveOpBatched_case0(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((11, 11)).astype(self.dtype), 'Y': np.random.random((2, 11, 7)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.1, check_pir=True)

class TestSolveOpBatched_case1(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((20, 6, 6)).astype(self.dtype), 'Y': np.random.random((20, 6)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.04, check_pir=True)

class TestSolveOpBatched_case2(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((2, 10, 10)).astype(self.dtype), 'Y': np.random.random((1, 10, 10)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.02, check_pir=True)

class TestSolveOpBatched_case3(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((1, 10, 10)).astype(self.dtype), 'Y': np.random.random((2, 10, 10)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.02, check_pir=True)

class TestSolveOpBatched_case4(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((3, 6, 6)).astype(self.dtype), 'Y': np.random.random((3, 6, 7)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

class TestSolveOpBatched_case5(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((2, 2, 6, 6)).astype(self.dtype), 'Y': np.random.random((2, 2, 6, 6)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

class TestSolveOpBatched_case6(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((2, 2, 6, 6)).astype(self.dtype), 'Y': np.random.random((1, 2, 6, 9)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            return 10
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

class TestSolveOpBatched_case7(OpTest):

    def setUp(self):
        if False:
            return 10
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype), 'Y': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.04, check_pir=True)

class TestSolveOpBatched_case8(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = paddle.linalg.solve
        self.op_type = 'solve'
        self.dtype = 'float64'
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype), 'Y': np.random.random((1, 2, 2, 4, 7)).astype(self.dtype)}
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.04, check_pir=True)

class TestSolveOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        with program_guard(Program(), Program()):
            x1 = base.create_lod_tensor(np.array([[-1]]), [[1]], base.CPUPlace())
            y1 = base.create_lod_tensor(np.array([[-1]]), [[1]], base.CPUPlace())
            self.assertRaises(TypeError, paddle.linalg.solve, x1, y1)
            x2 = paddle.static.data(name='x2', shape=[30, 30], dtype='bool')
            y2 = paddle.static.data(name='y2', shape=[30, 10], dtype='bool')
            self.assertRaises(TypeError, paddle.linalg.solve, x2, y2)
            x3 = paddle.static.data(name='x3', shape=[30, 30], dtype='int32')
            y3 = paddle.static.data(name='y3', shape=[30, 10], dtype='int32')
            self.assertRaises(TypeError, paddle.linalg.solve, x3, y3)
            x4 = paddle.static.data(name='x4', shape=[30, 30], dtype='int64')
            y4 = paddle.static.data(name='y4', shape=[30, 10], dtype='int64')
            self.assertRaises(TypeError, paddle.linalg.solve, x4, y4)
            x5 = paddle.static.data(name='x5', shape=[30, 30], dtype='float16')
            y5 = paddle.static.data(name='y5', shape=[30, 10], dtype='float16')
            self.assertRaises(TypeError, paddle.linalg.solve, x5, y5)
            x6 = paddle.static.data(name='x6', shape=[30], dtype='float64')
            y6 = paddle.static.data(name='y6', shape=[30], dtype='float64')
            self.assertRaises(ValueError, paddle.linalg.solve, x6, y6)
            x7 = paddle.static.data(name='x7', shape=[2, 3, 4], dtype='float64')
            y7 = paddle.static.data(name='y7', shape=[2, 4, 3], dtype='float64')
            self.assertRaises(ValueError, paddle.linalg.solve, x7, y7)

class TestSolveOpAPI_1(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = 'float64'
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(name='input_x', shape=[3, 3], dtype=self.dtype)
            paddle_input_y = paddle.static.data(name='input_y', shape=[3], dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)
            np_input_x = np.random.random([3, 3]).astype(self.dtype)
            np_input_y = np.random.random([3]).astype(self.dtype)
            np_result = np.linalg.solve(np_input_x, np_input_y)
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'input_x': np_input_x, 'input_y': np_input_y}, fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=1e-05)

    def test_static(self):
        if False:
            i = 10
            return i + 15
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            i = 10
            return i + 15

        def run(place):
            if False:
                print('Hello World!')
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([3, 3]).astype(self.dtype)
            input_y_np = np.random.random([3]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)
            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()
        for place in self.place:
            run(place)

class TestSolveOpAPI_2(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = 'float64'
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(name='input_x', shape=[10, 10], dtype=self.dtype)
            paddle_input_y = paddle.static.data(name='input_y', shape=[10, 4], dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)
            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([10, 4]).astype(self.dtype)
            np_result = np.linalg.solve(np_input_x, np_input_y)
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'input_x': np_input_x, 'input_y': np_input_y}, fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=1e-05)

    def test_static(self):
        if False:
            while True:
                i = 10
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            print('Hello World!')

        def run(place):
            if False:
                return 10
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([10, 4]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)
            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()
        for place in self.place:
            run(place)

class TestSolveOpAPI_3(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = 'float32'
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(name='input_x', shape=[10, 10], dtype=self.dtype)
            paddle_input_y = paddle.static.data(name='input_y', shape=[10, 4], dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)
            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([10, 4]).astype(self.dtype)
            np_result = np.linalg.solve(np_input_x, np_input_y)
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'input_x': np_input_x, 'input_y': np_input_y}, fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=0.0001)

    def test_static(self):
        if False:
            print('Hello World!')
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')

        def run(place):
            if False:
                print('Hello World!')
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([10, 4]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)
            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=0.0001)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()
        for place in self.place:
            run(place)

class TestSolveOpAPI_4(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = 'float64'
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            return 10
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(name='input_x', shape=[2, 3, 3], dtype=self.dtype)
            paddle_input_y = paddle.static.data(name='input_y', shape=[1, 3, 3], dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)
            np_input_x = np.random.random([2, 3, 3]).astype(self.dtype)
            np_input_y = np.random.random([1, 3, 3]).astype(self.dtype)
            np_result = np.linalg.solve(np_input_x, np_input_y)
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'input_x': np_input_x, 'input_y': np_input_y}, fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=1e-05)

    def test_static(self):
        if False:
            i = 10
            return i + 15
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')

        def run(place):
            if False:
                for i in range(10):
                    print('nop')
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([2, 3, 3]).astype(self.dtype)
            input_y_np = np.random.random([1, 3, 3]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)
            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()
        for place in self.place:
            run(place)

class TestSolveOpSingularAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.places = [base.CPUPlace()]
        self.dtype = 'float64'
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            while True:
                i = 10
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.static.data(name='x', shape=[4, 4], dtype=self.dtype)
            y = paddle.static.data(name='y', shape=[4, 4], dtype=self.dtype)
            result = paddle.linalg.solve(x, y)
            input_x_np = np.ones([4, 4]).astype(self.dtype)
            input_y_np = np.ones([4, 4]).astype(self.dtype)
            exe = base.Executor(place)
            try:
                fetches = exe.run(base.default_main_program(), feed={'x': input_x_np, 'y': input_y_np}, fetch_list=[result])
            except RuntimeError as ex:
                print('The mat is singular')
            except ValueError as ex:
                print('The mat is singular')

    def test_static(self):
        if False:
            return 10
        for place in self.places:
            paddle.enable_static()
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            print('Hello World!')
        for place in self.places:
            with base.dygraph.guard(place):
                input_x_np = np.ones([4, 4]).astype(self.dtype)
                input_y_np = np.ones([4, 4]).astype(self.dtype)
                input_x = base.dygraph.to_variable(input_x_np)
                input_y = base.dygraph.to_variable(input_y_np)
                try:
                    result = paddle.linalg.solve(input_x, input_y)
                except RuntimeError as ex:
                    print('The mat is singular')
                except ValueError as ex:
                    print('The mat is singular')
if __name__ == '__main__':
    unittest.main()