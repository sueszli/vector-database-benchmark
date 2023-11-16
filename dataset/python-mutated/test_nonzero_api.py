import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import base
from paddle.base import Program, program_guard

def call_nonzero(x):
    if False:
        return 10
    input = paddle.to_tensor(x)
    return paddle.nonzero(x=input)

class TestNonZeroAPI(unittest.TestCase):

    def test_nonzero_api_as_tuple(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 2)
            z = paddle.concat(list(y), axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(feed={'x': data}, fetch_list=[z.name], return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)
        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1], dtype='float32')
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 1)
            z = paddle.concat(list(y), axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(feed={'x': data}, fetch_list=[z.name], return_numpy=False)
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_nonzero_api(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(feed={'x': data}, fetch_list=[y.name], return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)
        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1], dtype='float32')
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(feed={'x': data}, fetch_list=[y.name], return_numpy=False)
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        if False:
            while True:
                i = 10
        data_x = np.array([[True, False], [False, True]])
        with base.dygraph.guard():
            x = base.dygraph.to_variable(data_x)
            z = paddle.nonzero(x)
            np_z = z.numpy()
        expect_out = np.array([[0, 0], [1, 1]])

class TestNonzeroOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        'Test where_index op with random value'
        np.random.seed(2023)
        self.op_type = 'where_index'
        self.python_api = call_nonzero
        self.init_shape()
        self.init_dtype()
        self.inputs = self.create_inputs()
        self.outputs = self.return_outputs()

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)

    def init_shape(self):
        if False:
            return 10
        self.shape = [8, 8]

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float64

    def create_inputs(self):
        if False:
            print('Hello World!')
        return {'Condition': np.random.randint(5, size=self.shape).astype(self.dtype)}

    def return_outputs(self):
        if False:
            print('Hello World!')
        return {'Out': np.transpose(np.nonzero(self.inputs['Condition']))}

class TestNonzeroFP32Op(TestNonzeroOp):

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = [2, 10, 2]

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float32

class TestNonzeroFP16Op(TestNonzeroOp):

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [3, 4, 7]

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

class TestNonzeroBF16(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Test where_index op with bfloat16 dtype'
        np.random.seed(2023)
        self.op_type = 'where_index'
        self.python_api = call_nonzero
        self.init_shape()
        self.init_dtype()
        self.inputs = self.create_inputs()
        self.outputs = self.return_outputs()

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = [12, 9]

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.uint16

    def create_inputs(self):
        if False:
            while True:
                i = 10
        return {'Condition': convert_float_to_uint16(np.random.randint(5, size=self.shape).astype(np.float32))}

    def return_outputs(self):
        if False:
            while True:
                i = 10
        return {'Out': np.transpose(np.nonzero(self.inputs['Condition']))}
if __name__ == '__main__':
    unittest.main()