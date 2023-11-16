import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
paddle.enable_static()

class XPUTestInverseOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'inverse'
        self.use_dynamic_create_class = False

    class TestXPUInverseOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.op_type = 'inverse'
            self.place = paddle.XPUPlace(0)
            self.set_dtype()
            self.set_shape()
            self.init_input_output()

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.input_shape = [10, 10]

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            np.random.seed(123)
            x = np.random.random(self.input_shape).astype(self.dtype)
            out = np.linalg.inv(x).astype(self.dtype)
            self.inputs = {'Input': x}
            self.outputs = {'Output': out}

        def set_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(self.place, ['Input'], 'Output')

    class TestXPUInverseOpBatched(TestXPUInverseOp):

        def set_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.input_shape = [8, 4, 4]

    class TestXPUInverseOpLarge(TestXPUInverseOp):

        def set_shape(self):
            if False:
                return 10
            self.input_shape = [32, 32]
support_types = get_xpu_op_support_types('inverse')
for stype in support_types:
    create_test_class(globals(), XPUTestInverseOp, stype)

class TestInverseSingularAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.places = [base.XPUPlace(0)]

    def check_static_result(self, place):
        if False:
            while True:
                i = 10
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name='input', shape=[4, 4], dtype='float32')
            result = paddle.inverse(x=input)
            input_np = np.ones([4, 4]).astype('float32')
            exe = base.Executor(place)
            with self.assertRaises(OSError):
                fetches = exe.run(base.default_main_program(), feed={'input': input_np}, fetch_list=[result])

    def test_static(self):
        if False:
            return 10
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            print('Hello World!')
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.ones([4, 4]).astype('float32')
                input = base.dygraph.to_variable(input_np)
                with self.assertRaises(OSError):
                    result = paddle.inverse(input)
if __name__ == '__main__':
    unittest.main()