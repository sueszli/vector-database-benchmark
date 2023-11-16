import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
from paddle.base import framework
paddle.enable_static()

class XPUTestAssignValueOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'assign_value'
        self.use_dynamic_create_class = False

    class TestAssignValueOp(XPUOpTest):

        def init(self):
            if False:
                print('Hello World!')
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = 'assign_value'

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.init()
            self.inputs = {}
            self.attrs = {}
            self.init_data()
            self.attrs['shape'] = self.value.shape
            self.attrs['dtype'] = framework.convert_np_dtype_to_dtype_(self.value.dtype)
            self.outputs = {'Out': self.value}

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.value = np.random.random(size=(2, 5)).astype(np.float32)
            self.attrs['fp32_values'] = [float(v) for v in self.value.flat]

        def test_forward(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

    class TestAssignValueOp2(TestAssignValueOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.value = np.random.random(size=(2, 5)).astype(np.int32)
            self.attrs['int32_values'] = [int(v) for v in self.value.flat]

    class TestAssignValueOp3(TestAssignValueOp):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.value = np.random.random(size=(2, 5)).astype(np.int64)
            self.attrs['int64_values'] = [int(v) for v in self.value.flat]

    class TestAssignValueOp4(TestAssignValueOp):

        def init_data(self):
            if False:
                print('Hello World!')
            self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
            self.attrs['bool_values'] = [int(v) for v in self.value.flat]

class TestAssignApi(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_dtype()
        self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(self.dtype)
        self.place = base.XPUPlace(0)

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = 'float32'

    def test_assign(self):
        if False:
            i = 10
            return i + 15
        main_program = base.Program()
        with base.program_guard(main_program):
            x = paddle.tensor.create_tensor(dtype=self.dtype)
            paddle.assign(self.value, output=x)
        exe = base.Executor(self.place)
        [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
        np.testing.assert_allclose(fetched_x, self.value)
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
            for i in range(10):
                print('nop')
        self.dtype = 'int64'

class TestAssignApi4(TestAssignApi):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_dtype()
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
        self.place = base.XPUPlace(0)

    def init_dtype(self):
        if False:
            return 10
        self.dtype = 'bool'
support_types = get_xpu_op_support_types('assign_value')
for stype in support_types:
    create_test_class(globals(), XPUTestAssignValueOp, stype)
if __name__ == '__main__':
    unittest.main()