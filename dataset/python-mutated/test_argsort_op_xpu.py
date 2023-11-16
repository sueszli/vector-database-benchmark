import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestArgsortOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'argsort'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        if False:
            while True:
                i = 10
        base_class = self.TestArgsortOp
        classes = []
        for descending in [True, False]:
            for axis in [0, 1, 2, -1, -2]:
                class_name = 'XPUTestArgsortOp_axis_' + str(axis) + '_' + str(descending)
                attr_dict = {'init_axis': axis, 'init_descending': descending}
                classes.append([class_name, attr_dict])
        return (base_class, classes)

    class TestArgsortOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.set_xpu()
            self.op_type = 'argsort'
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.input_shape = (2, 2, 2, 3, 3)
            self.axis = -1 if not hasattr(self, 'init_axis') else self.init_axis
            self.descending = False if not hasattr(self, 'init_descending') else self.init_descending
            if self.dtype == np.float32:
                self.x = np.random.random(self.input_shape).astype(self.dtype)
            else:
                self.x = np.random.randint(low=-1000, high=1000, size=self.input_shape).astype(self.dtype)
            self.inputs = {'X': self.x}
            self.attrs = {'axis': self.axis, 'descending': self.descending}
            self.get_output()
            self.outputs = {'Out': self.sorted_x, 'Indices': self.indices}

        def get_output(self):
            if False:
                print('Hello World!')
            if self.descending:
                self.indices = np.flip(np.argsort(self.x, kind='heapsort', axis=self.axis), self.axis)
                self.sorted_x = np.flip(np.sort(self.x, kind='heapsort', axis=self.axis), self.axis)
            else:
                self.indices = np.argsort(self.x, kind='heapsort', axis=self.axis)
                self.sorted_x = np.sort(self.x, kind='heapsort', axis=self.axis)

        def set_xpu(self):
            if False:
                while True:
                    i = 10
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(self.place, {'X'}, 'Out')

class XPUTestArgsortOp_LargeN(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'argsort'
        self.use_dynamic_create_class = False

    class TestArgsortOpCase1(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.set_xpu()
            self.op_type = 'argsort'
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.axis = -1 if not hasattr(self, 'init_axis') else self.init_axis
            self.init_test_case()
            self.descending = False if not hasattr(self, 'init_descending') else self.init_descending
            np.random.seed(100)
            if self.dtype == np.float32:
                self.x = np.random.random(self.input_shape).astype(self.dtype)
            else:
                self.x = np.random.choice(1000000, self.input_shape, replace=False).astype(self.dtype)
            self.inputs = {'X': self.x}
            self.attrs = {'axis': self.axis, 'descending': self.descending}
            self.get_output()
            self.outputs = {'Out': self.sorted_x, 'Indices': self.indices}

        def get_output(self):
            if False:
                i = 10
                return i + 15
            if self.descending:
                self.indices = np.flip(np.argsort(self.x, kind='heapsort', axis=self.axis), self.axis)
                self.sorted_x = np.flip(np.sort(self.x, kind='heapsort', axis=self.axis), self.axis)
            else:
                self.indices = np.argsort(self.x, kind='heapsort', axis=self.axis)
                self.sorted_x = np.sort(self.x, kind='heapsort', axis=self.axis)

        def set_xpu(self):
            if False:
                for i in range(10):
                    print('nop')
            self.__class__.use_xpu = True

        def init_test_case(self):
            if False:
                return 10
            self.input_shape = [2, 8732]

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(self.place, {'X'}, 'Out')

    class TestArgsortOpCase2(TestArgsortOpCase1):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.input_shape = [2, 10241]

    class TestArgsortOpCase3(TestArgsortOpCase1):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.input_shape = [2, 8732, 1]
            self.axis = 1

    class TestArgsortOpCase4(TestArgsortOpCase1):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.input_shape = [2, 10241, 1]
            self.axis = 1
support_types = get_xpu_op_support_types('argsort')
for stype in support_types:
    create_test_class(globals(), XPUTestArgsortOp, stype)
    if stype != 'float16':
        create_test_class(globals(), XPUTestArgsortOp_LargeN, stype)
if __name__ == '__main__':
    unittest.main()