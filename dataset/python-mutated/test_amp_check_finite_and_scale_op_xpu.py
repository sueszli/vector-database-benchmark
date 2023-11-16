import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestCheckFiniteAndUnscaleOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'check_finite_and_unscale'
        self.use_dynamic_create_class = False

    class TestCheckFiniteAndUnscaleOpNormal(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'check_finite_and_unscale'
            self.init_dtype()
            x = np.random.random((8, 8)).astype(self.dtype)
            scale = np.random.random(1).astype(np.float32)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {'FoundInfinite': np.array([0]), 'Out': [('out0', x / scale)]}

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                print('Hello World!')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestCheckFiniteAndUnscaleOpWithNan(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'check_finite_and_unscale'
            self.init_dtype()
            x = np.random.random((256, 256)).astype(self.dtype)
            idx1 = np.random.randint(255)
            idx2 = np.random.randint(255)
            x[idx1][idx2] = np.nan
            x[idx2][idx1] = np.nan
            scale = np.random.random(1).astype(np.float32)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {'FoundInfinite': np.array([1]), 'Out': [('out0', x)]}

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['Out'])

    class TestCheckFiniteAndUnscaleOpWithInf(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'check_finite_and_unscale'
            self.init_dtype()
            x = np.random.random((256, 256)).astype(self.dtype)
            idx1 = np.random.randint(255)
            idx2 = np.random.randint(255)
            x[idx1][idx2] = np.nan
            x[idx2][idx1] = np.nan
            scale = np.random.random(1).astype(np.float32)
            myscale = np.array([0.05]).astype(self.dtype)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {'FoundInfinite': np.array([1]), 'Out': [('out0', x)]}

        def init_dtype(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['Out'])

    class TestCheckFiniteAndUnscaleOpWithInfAndNan(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'check_finite_and_unscale'
            self.init_dtype()
            x = np.random.random((256, 256)).astype(self.dtype)
            idx1 = np.random.randint(255)
            idx2 = np.random.randint(255)
            x[idx1][idx2] = np.inf
            x[idx2][idx1] = np.nan
            scale = np.random.random(1).astype(np.float32)
            myscale = np.array([0.05]).astype(self.dtype)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {'FoundInfinite': np.array([1]), 'Out': [('out0', x)]}

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                print('Hello World!')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['Out'])
support_types = get_xpu_op_support_types('check_finite_and_unscale')
for stype in support_types:
    create_test_class(globals(), XPUTestCheckFiniteAndUnscaleOp, stype)
if __name__ == '__main__':
    unittest.main()