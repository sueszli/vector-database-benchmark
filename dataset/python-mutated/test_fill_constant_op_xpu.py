import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest
import paddle

class XPUTestFillConstantOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'fill_constant'
        self.use_dynamic_create_class = False

    class TestFillConstantOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            'Test fill_constant op with specified value'
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'fill_constant'
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.convert_dtype2index()
            self.set_value()
            self.set_data()

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def set_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = [90, 10]

        def set_xpu(self):
            if False:
                return 10
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def convert_dtype2index(self):
            if False:
                print('Hello World!')
            '\n            if new type added, need to add corresponding index\n            '
            if self.dtype == np.bool_:
                self.index = 0
            if self.dtype == np.int16:
                self.index = 1
            if self.dtype == np.int32:
                self.index = 2
            if self.dtype == np.int64:
                self.index = 3
            if self.dtype == np.float16:
                self.index = 4
            if self.dtype == np.float32:
                self.index = 5
            if self.dtype == np.float64:
                self.index = 6
            if self.dtype == np.uint8:
                self.index = 20
            if self.dtype == np.int8:
                self.index = 21
            if self.dtype == np.uint16:
                self.index = 22
            if self.dtype == np.complex64:
                self.index = 23
            if self.dtype == np.complex128:
                self.index = 24

        def set_value(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.index == 3:
                self.value = 10000000000
            elif self.index == 0:
                self.value = np.random.randint(0, 2)
            elif self.index in [20, 21]:
                self.value = 125
            elif self.index in [1, 2]:
                self.value = 7
            elif self.index in [4, 5, 6]:
                self.value = 1e-05
            elif self.index == 22:
                self.value = 1.0
            else:
                self.value = 3.7

        def set_data(self):
            if False:
                print('Hello World!')
            self.inputs = {}
            self.attrs = {'shape': self.shape, 'dtype': self.index, 'value': self.value}
            self.outputs = {'Out': np.full(self.shape, self.value)}

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

    class TestFillConstantOp2(TestFillConstantOp):
        """Test fill_constant op with default value"""

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = [10, 10]

    class TestFillConstantOp3(TestFillConstantOp):
        """Test fill_constant op with specified int64 value"""

        def set_shape(self):
            if False:
                print('Hello World!')
            self.shape = [123, 2, 1]

    class TestFillConstantOp4(TestFillConstantOp):
        """Test fill_constant op with specified int value"""

        def set_shape(self):
            if False:
                return 10
            self.shape = [123, 3, 2, 1]

    class TestFillConstantOp5(TestFillConstantOp):
        """Test fill_constant op with specified float value"""

        def set_shape(self):
            if False:
                print('Hello World!')
            self.shape = [123]

    class TestFillConstantOp1_ShapeTensorList(TestFillConstantOp):
        """Test fill_constant op with specified value"""

        def set_data(self):
            if False:
                for i in range(10):
                    print('nop')
            shape_tensor_list = []
            for (index, ele) in enumerate(self.shape):
                shape_tensor_list.append(('x' + str(index), np.ones(1).astype('int32') * ele))
            self.inputs = {'ShapeTensorList': shape_tensor_list}
            self.attrs = {'shape': self.infer_shape, 'dtype': self.index, 'value': self.value}
            self.outputs = {'Out': np.full(self.shape, self.value)}
            if self.index == 22:
                self.outputs = {'Out': np.full(self.shape, convert_float_to_uint16(np.array([self.value]).astype('float32')))}

        def set_shape(self):
            if False:
                while True:
                    i = 10
            self.shape = [123, 92]
            self.infer_shape = [123, 1]

    class TestFillConstantOp2_ShapeTensorList(TestFillConstantOp):
        """Test fill_constant op with default value"""

        def set_data(self):
            if False:
                for i in range(10):
                    print('nop')
            shape_tensor_list = []
            for (index, ele) in enumerate(self.shape):
                shape_tensor_list.append(('x' + str(index), np.ones(1).astype('int32') * ele))
            self.inputs = {'ShapeTensorList': shape_tensor_list}
            self.attrs = {'shape': self.infer_shape, 'dtype': self.index}
            self.outputs = {'Out': np.full(self.shape, 0.0)}

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = [123, 2, 1]
            self.infer_shape = [1, 1, 1]

    class TestFillConstantOp3_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):

        def set_shape(self):
            if False:
                print('Hello World!')
            self.shape = [123, 3, 2, 1]
            self.infer_shape = [123, 111, 11, 1]

    class TestFillConstantOp4_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):

        def set_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = [123]
            self.infer_shape = [1]

    class TestFillConstantOp1_ShapeTensor(TestFillConstantOp):
        """Test fill_constant op with specified value"""

        def set_data(self):
            if False:
                i = 10
                return i + 15
            self.inputs = {'ShapeTensor': np.array(self.shape).astype('int32')}
            self.attrs = {'value': self.value, 'dtype': self.index}
            self.outputs = {'Out': np.full(self.shape, self.value)}
            if self.index == 22:
                self.outputs = {'Out': np.full(self.shape, convert_float_to_uint16(np.array([self.value]).astype('float32')))}

        def set_shape(self):
            if False:
                print('Hello World!')
            self.shape = [123, 92]

    class TestFillConstantOp1_ValueTensor(TestFillConstantOp):
        """Test fill_constant op with specified value"""

        def set_data(self):
            if False:
                i = 10
                return i + 15
            self.inputs = {'ShapeTensor': np.array(self.shape).astype('int32'), 'ValueTensor': np.array([self.value]).astype(self.dtype)}
            if self.index == 22:
                self.inputs = {'ValueTensor': convert_float_to_uint16(np.array([self.value]).astype('float32'))}
            self.attrs = {'value': self.value, 'dtype': self.index}
            self.outputs = {'Out': np.full(self.shape, self.value)}

        def set_shape(self):
            if False:
                while True:
                    i = 10
            self.shape = [123, 92]
support_types = get_xpu_op_support_types('fill_constant')
for stype in support_types:
    create_test_class(globals(), XPUTestFillConstantOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()