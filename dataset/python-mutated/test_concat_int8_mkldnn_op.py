import unittest
import numpy as np
from op_test import OpTest

class TestConcatOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'concat'
        self.use_mkldnn = True
        self._cpu_only = True
        self.init_axis()
        self.init_shape()
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}
        self.output = np.concatenate((self.x0, self.x1, self.x2), axis=self.axis).astype('int')
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_dygraph=False)

    def init_test_data(self):
        if False:
            return 10
        self.x0 = (np.random.randint(0, 100, self.x0_shape) - 50).astype('int8')
        self.x1 = (np.random.randint(0, 80, self.x1_shape) - 30).astype('int8')
        self.x2 = (np.random.randint(0, 110, self.x2_shape) - 80).astype('int8')

    def init_axis(self):
        if False:
            print('Hello World!')
        self.axis = 0

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x0_shape = [2, 2, 1, 2]
        self.x1_shape = [1, 2, 1, 2]
        self.x2_shape = [3, 2, 1, 2]

class TestConcatOp2(TestConcatOp):

    def init_test_data(self):
        if False:
            i = 10
            return i + 15
        self.x0 = np.random.randint(0, 100, self.x0_shape).astype('uint8')
        self.x1 = np.random.randint(0, 50, self.x1_shape).astype('uint8')
        self.x2 = np.random.randint(0, 80, self.x2_shape).astype('uint8')

    def init_axis(self):
        if False:
            while True:
                i = 10
        self.axis = 0

    def init_shape(self):
        if False:
            return 10
        self.x0_shape = [2, 1, 5, 5]
        self.x1_shape = [1, 1, 5, 5]
        self.x2_shape = [3, 1, 5, 5]

def create_test_int8_class(parent):
    if False:
        i = 10
        return i + 15

    class TestAxis1Case(parent):

        def init_axis(self):
            if False:
                i = 10
                return i + 15
            self.axis = 1

        def init_shape(self):
            if False:
                print('Hello World!')
            self.x0_shape = [1, 1, 5, 5]
            self.x1_shape = [1, 2, 5, 5]
            self.x2_shape = [1, 3, 5, 5]

    class TestAxis2Case(parent):

        def init_axis(self):
            if False:
                i = 10
                return i + 15
            self.axis = 2

        def init_shape(self):
            if False:
                i = 10
                return i + 15
            self.x0_shape = [2, 3, 4, 5]
            self.x1_shape = [2, 3, 5, 5]
            self.x2_shape = [2, 3, 6, 5]

    class TestAxis3Case(parent):

        def init_axis(self):
            if False:
                while True:
                    i = 10
            self.axis = 3

        def init_shape(self):
            if False:
                return 10
            self.x0_shape = [2, 3, 5, 5]
            self.x1_shape = [2, 3, 5, 6]
            self.x2_shape = [2, 3, 5, 7]
    cls_name_1 = '{}_axis_{}'.format(parent.__name__, '1')
    cls_name_2 = '{}_axis_{}'.format(parent.__name__, '2')
    cls_name_3 = '{}_axis_{}'.format(parent.__name__, '3')
    TestAxis1Case.__name__ = cls_name_1
    TestAxis2Case.__name__ = cls_name_2
    TestAxis3Case.__name__ = cls_name_3
    globals()[cls_name_1] = TestAxis1Case
    globals()[cls_name_2] = TestAxis2Case
    globals()[cls_name_3] = TestAxis3Case
create_test_int8_class(TestConcatOp)
create_test_int8_class(TestConcatOp2)
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()