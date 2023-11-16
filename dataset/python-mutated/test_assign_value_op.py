import unittest
import numpy as np
import op_test
import paddle
from paddle import base
from paddle.base import framework

def assign_value_wrapper(shape=[], dtype=base.core.VarDesc.VarType.FP32, values=0.0):
    if False:
        while True:
            i = 10
    if paddle.framework.in_dynamic_mode():
        tensor = paddle.Tensor()
    else:
        np_type = paddle.base.data_feeder._PADDLE_DTYPE_2_NUMPY_DTYPE[dtype]
        tensor = paddle.zeros(list(shape), np_type)
        dtype = paddle.pir.core.convert_np_dtype_to_dtype_(np_type)
    return paddle._C_ops.assign_value_(tensor, shape, dtype, values, framework._current_expected_place())

class TestAssignValueOp(op_test.OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'assign_value'
        self.python_api = assign_value_wrapper
        self.inputs = {}
        self.attrs = {}
        self.init_data()
        self.attrs['shape'] = self.value.shape
        self.attrs['dtype'] = framework.convert_np_dtype_to_dtype_(self.value.dtype)
        self.outputs = {'Out': self.value}

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.value = np.random.random(size=(2, 5)).astype(np.float32)
        self.attrs['fp32_values'] = [float(v) for v in self.value.flat]

    def test_forward(self):
        if False:
            return 10
        self.check_output(check_cinn=True, check_pir=True)

class TestAssignValueOp2(TestAssignValueOp):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.value = np.random.random(size=(2, 5)).astype(np.int32)
        self.attrs['int32_values'] = [int(v) for v in self.value.flat]

class TestAssignValueOp3(TestAssignValueOp):

    def init_data(self):
        if False:
            return 10
        self.value = np.random.random(size=(2, 5)).astype(np.int64)
        self.attrs['int64_values'] = [int(v) for v in self.value.flat]

class TestAssignValueOp4(TestAssignValueOp):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
        self.attrs['bool_values'] = [int(v) for v in self.value.flat]

class TestAssignApi(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with op_test.paddle_static_guard():
            self.init_dtype()
            self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(self.dtype)
            self.place = base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'float32'

    def test_assign(self):
        if False:
            while True:
                i = 10
        with op_test.paddle_static_guard():
            main_program = base.Program()
            with base.program_guard(main_program):
                x = paddle.tensor.create_tensor(dtype=self.dtype)
                paddle.assign(self.value, output=x)
            exe = base.Executor(self.place)
            [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
            np.testing.assert_array_equal(fetched_x, self.value)
            self.assertEqual(fetched_x.dtype, self.value.dtype)

    def test_pir_assign(self):
        if False:
            print('Hello World!')
        with paddle.pir_utils.IrGuard():
            main_program = paddle.pir.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.zeros(shape=[1], dtype=self.dtype)
                paddle.assign(self.value, output=x)
            exe = base.Executor(self.place)
            [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
            np.testing.assert_array_equal(fetched_x, self.value)
            self.assertEqual(fetched_x.dtype, self.value.dtype)

class TestAssignApi2(TestAssignApi):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'int32'

class TestAssignApi3(TestAssignApi):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = 'int64'

class TestAssignApi4(TestAssignApi):

    def setUp(self):
        if False:
            return 10
        with op_test.paddle_static_guard():
            self.init_dtype()
            self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
            self.place = base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'bool'
if __name__ == '__main__':
    unittest.main()