import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReshapeOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'reshape2'
        self.use_dynamic_create_class = False

    class TestReshapeOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.init_data()
            self.op_type = 'reshape2'
            self.dtype = self.in_type
            self.init_test_input()
            self.init_test_output()
            self.init_attrs()

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.ori_shape = (2, 60)
            self.new_shape = (12, 10)
            self.infered_shape = (12, 10)

        def init_test_input(self):
            if False:
                return 10
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype)}

        def init_test_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}

        def init_attrs(self):
            if False:
                return 10
            self.attrs = {'shape': self.new_shape, 'use_xpu': True}

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X'], 'Out')

    class TestReshapeOpDimInfer1(TestReshapeOp):

        def init_data(self):
            if False:
                return 10
            self.ori_shape = (5, 25)
            self.new_shape = (5, -1, 5)
            self.infered_shape = (5, -1, 5)

    class TestReshapeOpDimInfer2(TestReshapeOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (10, 2, 6)
            self.new_shape = (10, 0, 3, -1)
            self.infered_shape = (10, 2, 3, -1)

    class TestReshapeOpWithInputShape(TestReshapeOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (6, 20)
            self.new_shape = (0, -1, 20)
            self.actual_shape = (2, 3, 20)

        def init_test_input(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'Shape': np.array(self.actual_shape, dtype='int32')}

        def init_test_output(self):
            if False:
                print('Hello World!')
            self.outputs = {'Out': self.inputs['X'].reshape(self.actual_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}

    class TestReshapeOp_attr_ShapeTensor(TestReshapeOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (4, 25)
            self.new_shape = (10, 10)
            self.infered_shape = (10, 10)
            self.shape = (-1, -1)

        def init_test_input(self):
            if False:
                return 10
            shape_tensor = []
            for (index, ele) in enumerate(self.new_shape):
                shape_tensor.append(('x' + str(index), np.ones(1).astype('int32') * ele))
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'ShapeTensor': shape_tensor}

        def init_attrs(self):
            if False:
                return 10
            self.attrs = {'shape': self.shape, 'use_xpu': True}

    class TestReshapeOpDimInfer1_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (5, 20)
            self.new_shape = (5, -1, 20)
            self.infered_shape = (5, -1, 20)
            self.shape = (5, -1, -1)

    class TestReshapeOpDimInfer2_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.ori_shape = (10, 2, 6)
            self.new_shape = (10, 0, 3, -1)
            self.infered_shape = (10, 2, 3, -1)
            self.shape = (10, 0, 3, -1)

    class TestReshapeOp_attr_OnlyShape(TestReshapeOp):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (4, 25)
            self.new_shape = (10, 10)
            self.infered_shape = (10, 10)

        def init_test_input(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'Shape': np.array(self.new_shape, dtype='int32')}

        def init_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'use_xpu': True}

    class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):

        def init_data(self):
            if False:
                return 10
            self.ori_shape = (5, 20)
            self.new_shape = (5, -1, 10)
            self.infered_shape = (5, -1, 10)
            self.shape = (5, -1, -1)

    class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (10, 2, 6)
            self.new_shape = (10, 0, 3, -1)
            self.infered_shape = (10, 2, 3, -1)
            self.shape = (10, 0, 3, -1)
support_types = get_xpu_op_support_types('reshape2')
for stype in support_types:
    create_test_class(globals(), XPUTestReshapeOp, stype)
if __name__ == '__main__':
    unittest.main()