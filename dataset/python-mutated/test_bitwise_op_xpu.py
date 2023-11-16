import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestBitwiseAnd(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'bitwise_and'

    class XPUTestBitwiseAndBase(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'bitwise_and'
            if self.dtype == np.bool_:
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.bitwise_and(x, y)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x), 'Y': OpTest.np_dtype_to_base_dtype(y)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
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
                for i in range(10):
                    print('nop')
            pass

    class XPUTestBitwiseAndCase1(XPUTestBitwiseAndBase):

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseAndCase2(XPUTestBitwiseAndBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [4, 1]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseAndCase3(XPUTestBitwiseAndBase):

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = 0
            self.high = 100
support_types = get_xpu_op_support_types('bitwise_and')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseAnd, stype)

class XPUTestBitwiseOr(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'bitwise_or'

    class XPUTestBitwiseOrBase(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                return 10
            self.op_type = 'bitwise_or'
            if self.dtype == np.bool_:
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.bitwise_or(x, y)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x), 'Y': OpTest.np_dtype_to_base_dtype(y)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            pass

    class XPUTestBitwiseOrCase1(XPUTestBitwiseOrBase):

        def init_case(self):
            if False:
                return 10
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseOrCase2(XPUTestBitwiseOrBase):

        def init_case(self):
            if False:
                return 10
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [4, 1]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseOrCase3(XPUTestBitwiseOrBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = 0
            self.high = 100
support_types = get_xpu_op_support_types('bitwise_or')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseOr, stype)

class XPUTestBitwiseXor(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'bitwise_xor'

    class XPUTestBitwiseXorBase(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                return 10
            self.op_type = 'bitwise_xor'
            if self.dtype == np.bool_:
                self.low = 0
                self.high = 2
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.bitwise_xor(x, y)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x), 'Y': OpTest.np_dtype_to_base_dtype(y)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

    class XPUTestBitwiseXorCase1(XPUTestBitwiseXorBase):

        def init_case(self):
            if False:
                print('Hello World!')
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseXorCase2(XPUTestBitwiseXorBase):

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [4, 1]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseXorCase3(XPUTestBitwiseXorBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = 0
            self.high = 100
support_types = get_xpu_op_support_types('bitwise_xor')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseXor, stype)

class XPUTestBitwiseNot(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'bitwise_not'

    class XPUTestBitwiseNotBase(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'bitwise_not'
            x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
            out = np.bitwise_not(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                return 10
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            pass

    class XPUTestBitwiseNotBool(XPUTestBitwiseNotBase):

        def setUp(self):
            if False:
                print('Hello World!')
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'bitwise_not'
            x = np.random.choice([True, False], self.x_shape)
            out = np.bitwise_not(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': x}
            self.outputs = {'Out': out}

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.dtype = np.bool_
            self.x_shape = [2, 3, 4, 5]
support_types = get_xpu_op_support_types('bitwise_not')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseNot, stype)
if __name__ == '__main__':
    unittest.main()