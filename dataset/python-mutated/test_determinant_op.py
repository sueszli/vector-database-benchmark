import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

class TestDeterminantOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = paddle.linalg.det
        self.init_data()
        self.op_type = 'determinant'
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Input'], ['Out'], check_pir=True)

    def init_data(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        self.case = np.random.rand(3, 3, 3, 5, 5).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)

class TestDeterminantOpCase1(TestDeterminantOp):

    def init_data(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        self.case = np.random.rand(10, 10).astype('float32')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)

class TestDeterminantOpCase1FP16(TestDeterminantOp):

    def init_data(self):
        if False:
            return 10
        np.random.seed(0)
        self.case = np.random.rand(10, 10).astype(np.float16)
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case.astype(np.float32))

class TestDeterminantOpCase2(TestDeterminantOp):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(0)
        self.case = np.ones([4, 2, 4, 4]).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)

class TestDeterminantOpCase2FP16(TestDeterminantOp):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        self.case = np.ones([4, 2, 4, 4]).astype(np.float16)
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case.astype(np.float32)).astype(np.float16)

class TestDeterminantAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        np.random.seed(0)
        self.shape = [3, 3, 5, 5]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.place = paddle.CPUPlace()

    @test_with_pir_api
    def test_api_static(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.linalg.det(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.linalg.det(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.linalg.det(x_tensor)
        out_ref = np.linalg.det(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()

class TestSlogDeterminantOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'slogdeterminant'
        self.python_api = paddle.linalg.slogdet
        self.init_data()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['Input'], ['Out'], max_relative_error=0.1, check_pir=True)

    def init_data(self):
        if False:
            return 10
        np.random.seed(0)
        self.case = np.random.rand(4, 5, 5).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.array(np.linalg.slogdet(self.case))

class TestSlogDeterminantOpCase1(TestSlogDeterminantOp):

    def init_data(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        self.case = np.random.rand(2, 2, 5, 5).astype(np.float32)
        self.inputs = {'Input': self.case}
        self.target = np.array(np.linalg.slogdet(self.case))

class TestSlogDeterminantAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        self.shape = [3, 3, 5, 5]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.place = paddle.CPUPlace()

    @test_with_pir_api
    def test_api_static(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.linalg.slogdet(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.array(np.linalg.slogdet(self.x))
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.linalg.slogdet(x_tensor)
        out_ref = np.array(np.linalg.slogdet(self.x))
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()