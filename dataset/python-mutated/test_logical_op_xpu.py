import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestLogicalAnd(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'logical_and'

    class XPUTestLogicalAndBase(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'logical_and'
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.logical_and(x, y)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x), 'Y': OpTest.np_dtype_to_base_dtype(y)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                print('Hello World!')
            self.dtype = self.in_type
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            pass

    class XPUTestLogicalAndCase1(XPUTestLogicalAndBase):

        def init_case(self):
            if False:
                return 10
            self.dtype = self.in_type
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100
support_types = get_xpu_op_support_types('logical_and')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalAnd, stype)

class XPUTestLogicalOr(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'logical_or'

    class XPUTestLogicalOrBase(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'logical_or'
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.logical_or(x, y)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x), 'Y': OpTest.np_dtype_to_base_dtype(y)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                return 10
            pass

    class XPUTestLogicalOrCase1(XPUTestLogicalOrBase):

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100
support_types = get_xpu_op_support_types('logical_or')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalOr, stype)

class XPUTestLogicalXor(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'logical_xor'

    class XPUTestLogicalXorBase(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                return 10
            self.op_type = 'logical_xor'
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.logical_xor(x, y)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x), 'Y': OpTest.np_dtype_to_base_dtype(y)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                print('Hello World!')
            self.dtype = self.in_type
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

    class XPUTestLogicalXorCase1(XPUTestLogicalXorBase):

        def init_case(self):
            if False:
                return 10
            self.dtype = self.in_type
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100
support_types = get_xpu_op_support_types('logical_xor')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalXor, stype)

class XPUTestLogicalNot(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'logical_not'

    class XPUTestLogicalNotBase(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'logical_not'
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            out = np.logical_not(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type
            self.x_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
support_types = get_xpu_op_support_types('logical_not')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalNot, stype)
if __name__ == '__main__':
    unittest.main()