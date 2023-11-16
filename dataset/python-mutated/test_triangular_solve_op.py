import sys
import unittest
import numpy as np
sys.path.append('..')
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import Program, core, program_guard
paddle.enable_static()

class TestTriangularSolveOp(OpTest):
    """
    case 1
    """

    def config(self):
        if False:
            return 10
        self.x_shape = [12, 12]
        self.y_shape = [12, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            return 10
        self.output = np.linalg.solve(np.triu(self.inputs['X']), self.inputs['Y'])

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'triangular_solve'
        self.python_api = paddle.tensor.linalg.triangular_solve
        self.config()
        self.inputs = {'X': np.random.random(self.x_shape).astype(self.dtype), 'Y': np.random.random(self.y_shape).astype(self.dtype)}
        self.attrs = {'upper': self.upper, 'transpose': self.transpose, 'unitriangular': self.unitriangular}
        self.set_output()
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad_normal(self):
        if False:
            return 10
        self.check_grad(['X', 'Y'], 'Out', check_cinn=True, check_pir=True)

class TestTriangularSolveOp2(TestTriangularSolveOp):
    """
    case 2
    """

    def config(self):
        if False:
            return 10
        self.x_shape = [10, 10]
        self.y_shape = [3, 10, 8]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            return 10
        x = np.tril(self.inputs['X']).transpose(1, 0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

class TestTriangularSolveOp3(TestTriangularSolveOp):
    """
    case 3
    """

    def config(self):
        if False:
            print('Hello World!')
        self.x_shape = [1, 10, 10]
        self.y_shape = [6, 10, 12]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

class TestTriangularSolveOp4(TestTriangularSolveOp):
    """
    case 4
    """

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [3, 10, 10]
        self.y_shape = [1, 10, 12]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            print('Hello World!')
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

class TestTriangularSolveOp5(TestTriangularSolveOp):
    """
    case 5
    """

    def config(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = True
        self.dtype = 'float64'

    def set_output(self):
        if False:
            return 10
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.0)
        grad_out = np.ones([10, 10]).astype('float64')
        grad_y = np.linalg.solve(x.transpose(1, 0), grad_out)
        grad_x = -np.matmul(grad_y, self.output.transpose(1, 0))
        grad_x = np.triu(grad_x)
        np.fill_diagonal(grad_x, 0.0)
        self.check_grad(['X', 'Y'], 'Out', user_defined_grads=[grad_x, grad_y], user_defined_grad_outputs=[grad_out])

class TestTriangularSolveOp6(TestTriangularSolveOp):
    """
    case 6
    """

    def config(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [1, 3, 10, 10]
        self.y_shape = [2, 1, 10, 5]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            i = 10
            return i + 15
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

class TestTriangularSolveOp7(TestTriangularSolveOp):
    """
    case 7
    """

    def config(self):
        if False:
            print('Hello World!')
        self.x_shape = [2, 10, 10]
        self.y_shape = [5, 1, 10, 2]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            print('Hello World!')
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

class TestTriangularSolveOp8(TestTriangularSolveOp):
    """
    case 8
    """

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [12, 3, 3]
        self.y_shape = [2, 3, 12, 3, 2]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            return 10
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

class TestTriangularSolveOp9(TestTriangularSolveOp):
    """
    case 9
    """

    def config(self):
        if False:
            return 10
        self.x_shape = [2, 4, 2, 3, 3]
        self.y_shape = [4, 1, 3, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = 'float64'

    def set_output(self):
        if False:
            print('Hello World!')
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.matmul(np.linalg.inv(x), y)

class TestTriangularSolveAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = 'float64'
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            while True:
                i = 10
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.static.data(name='x', shape=[3, 3], dtype=self.dtype)
            y = paddle.static.data(name='y', shape=[3, 2], dtype=self.dtype)
            z = paddle.linalg.triangular_solve(x, y)
            x_np = np.random.random([3, 3]).astype(self.dtype)
            y_np = np.random.random([3, 2]).astype(self.dtype)
            z_np = np.linalg.solve(np.triu(x_np), y_np)
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'x': x_np, 'y': y_np}, fetch_list=[z])
            np.testing.assert_allclose(fetches[0], z_np, rtol=1e-05)

    def test_static(self):
        if False:
            while True:
                i = 10
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            return 10

        def run(place):
            if False:
                print('Hello World!')
            paddle.disable_static(place)
            x_np = np.random.random([3, 3]).astype(self.dtype)
            y_np = np.random.random([3, 2]).astype(self.dtype)
            z_np = np.linalg.solve(np.tril(x_np), y_np)
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            z = paddle.linalg.triangular_solve(x, y, upper=False)
            np.testing.assert_allclose(z_np, z.numpy(), rtol=1e-05)
            self.assertEqual(z_np.shape, z.numpy().shape)
            paddle.enable_static()
        for place in self.place:
            run(place)

class TestTriangularSolveOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):
            x1 = base.create_lod_tensor(np.array([[-1]]), [[1]], base.CPUPlace())
            y1 = base.create_lod_tensor(np.array([[-1]]), [[1]], base.CPUPlace())
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x1, y1)
            x2 = paddle.static.data(name='x2', shape=[30, 30], dtype='bool')
            y2 = paddle.static.data(name='y2', shape=[30, 10], dtype='bool')
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x2, y2)
            x3 = paddle.static.data(name='x3', shape=[30, 30], dtype='int32')
            y3 = paddle.static.data(name='y3', shape=[30, 10], dtype='int32')
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x3, y3)
            x4 = paddle.static.data(name='x4', shape=[30, 30], dtype='float16')
            y4 = paddle.static.data(name='y4', shape=[30, 10], dtype='float16')
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x4, y4)
            x5 = paddle.static.data(name='x5', shape=[30], dtype='float64')
            y5 = paddle.static.data(name='y5', shape=[30, 30], dtype='float64')
            self.assertRaises(ValueError, paddle.linalg.triangular_solve, x5, y5)
            x6 = paddle.static.data(name='x6', shape=[30, 30], dtype='float64')
            y6 = paddle.static.data(name='y6', shape=[30], dtype='float64')
            self.assertRaises(ValueError, paddle.linalg.triangular_solve, x6, y6)
            x7 = paddle.static.data(name='x7', shape=[2, 3, 4], dtype='float64')
            y7 = paddle.static.data(name='y7', shape=[2, 4, 3], dtype='float64')
            self.assertRaises(ValueError, paddle.linalg.triangular_solve, x7, y7)
if __name__ == '__main__':
    unittest.main()