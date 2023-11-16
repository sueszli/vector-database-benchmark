import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle.base.framework import convert_np_dtype_to_dtype_
paddle.enable_static()

class XPUTestEmptyOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'empty'
        self.use_dynamic_create_class = False

    class TestEmptyOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'empty'
            self.init_dtype()
            self.set_xpu()
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.set_inputs()
            self.init_config()

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_customized(self.verify_output)

        def verify_output(self, outs):
            if False:
                print('Hello World!')
            data_type = outs[0].dtype
            if data_type in ['float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'float16', 'int16']:
                max_value = np.nanmax(outs[0])
                min_value = np.nanmin(outs[0])
                always_full_zero = max_value == 0.0 and min_value == 0.0
                always_non_full_zero = max_value >= min_value
                self.assertTrue(always_full_zero or always_non_full_zero, 'always_full_zero or always_non_full_zero.')
            elif data_type in ['bool']:
                total_num = outs[0].size
                true_num = np.sum(outs[0])
                false_num = np.sum(~outs[0])
                self.assertTrue(total_num == true_num + false_num, 'The value should always be True or False.')
            else:
                self.assertTrue(False, 'invalid data type')

        def set_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = [500, 3]

        def set_inputs(self):
            if False:
                print('Hello World!')
            self.inputs = {}

        def init_config(self):
            if False:
                i = 10
                return i + 15
            dtype_inner = convert_np_dtype_to_dtype_(self.dtype)
            self.attrs = {'shape': self.shape, 'dtype': dtype_inner}
            self.outputs = {'Out': np.zeros(self.shape).astype(self.dtype)}

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def set_xpu(self):
            if False:
                print('Hello World!')
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.op_type

    class TestEmptyOpCase1(TestEmptyOp):

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = [50]

    class TestEmptyOpCase2(TestEmptyOp):

        def set_shape(self):
            if False:
                return 10
            self.shape = [1, 50, 3, 4]

    class TestEmptyOpCase3(TestEmptyOp):

        def set_shape(self):
            if False:
                while True:
                    i = 10
            self.shape = [5, 5, 5]

    class TestEmptyOp_ShapeTensor(TestEmptyOp):

        def set_inputs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'ShapeTensor': np.array(self.shape).astype('int32')}

    class TestEmptyOp_ShapeTensorList(TestEmptyOp):

        def set_inputs(self):
            if False:
                i = 10
                return i + 15
            shape_tensor_list = []
            for (index, ele) in enumerate(self.shape):
                shape_tensor_list.append(('x' + str(index), np.ones(1).astype('int32') * ele))
            self.inputs = {'ShapeTensorList': shape_tensor_list}
support_types = get_xpu_op_support_types('empty')
for stype in support_types:
    create_test_class(globals(), XPUTestEmptyOp, stype)
if __name__ == '__main__':
    unittest.main()