import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestUnsqueeze2Op(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'unsqueeze2'
        self.use_dynamic_create_class = False

    class TestUnsqueeze2Op(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'unsqueeze2'
            self.__class__.op_type = 'unsqueeze2'
            self.use_mkldnn = False
            self.init_dtype()
            self.init_test_case()
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'].reshape(self.new_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}
            self.init_attrs()

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def init_attrs(self):
            if False:
                return 10
            self.attrs = {'axes': self.axes}

        def init_test_case(self):
            if False:
                return 10
            self.ori_shape = (3, 40)
            self.axes = (1, 2)
            self.new_shape = (3, 1, 1, 40)

        def test_check_output(self):
            if False:
                return 10
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                return 10
            place = paddle.XPUPlace(0)
            if self.dtype == np.bool_:
                return
            else:
                self.check_grad_with_place(place, ['X'], 'Out')

    class TestUnsqueeze2Op1(TestUnsqueeze2Op):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (20, 5)
            self.axes = (-1,)
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2(TestUnsqueeze2Op):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    class TestUnsqueeze2Op3(TestUnsqueeze2Op):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    class TestUnsqueeze2Op4(TestUnsqueeze2Op):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)

    class TestUnsqueeze2Op_AxesTensorList(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'unsqueeze2'
            self.__class__.op_type = 'unsqueeze2'
            self.use_mkldnn = False
            self.init_dtype()
            self.init_test_case()
            axes_tensor_list = []
            for (index, ele) in enumerate(self.axes):
                axes_tensor_list.append(('axes' + str(index), np.ones(1).astype('int32') * ele))
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'AxesTensorList': axes_tensor_list}
            self.init_attrs()
            self.outputs = {'Out': self.inputs['X'].reshape(self.new_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                return 10
            place = paddle.XPUPlace(0)
            if self.dtype in [np.float32, np.float64, np.float16]:
                self.check_grad_with_place(place, ['X'], 'Out')
            else:
                return

        def init_test_case(self):
            if False:
                return 10
            self.ori_shape = (20, 5)
            self.axes = (1, 2)
            self.new_shape = (20, 1, 1, 5)

        def init_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {}

    class TestUnsqueeze2Op1_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.ori_shape = (20, 5)
            self.axes = (-1,)
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    class TestUnsqueeze2Op3_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    class TestUnsqueeze2Op4_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):

        def init_test_case(self):
            if False:
                return 10
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)

    class TestUnsqueeze2Op_AxesTensor(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'unsqueeze2'
            self.__class__.op_type = 'unsqueeze2'
            self.use_mkldnn = False
            self.init_test_case()
            self.init_dtype()
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'AxesTensor': np.array(self.axes).astype('int32')}
            self.init_attrs()
            self.outputs = {'Out': self.inputs['X'].reshape(self.new_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}

        def init_dtype(self):
            if False:
                print('Hello World!')
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                print('Hello World!')
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            place = paddle.XPUPlace(0)
            if self.dtype in [np.float32, np.float64, np.float16]:
                self.check_grad_with_place(place, ['X'], 'Out')
            else:
                return

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (20, 5)
            self.axes = (1, 2)
            self.new_shape = (20, 1, 1, 5)

        def init_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {}

    class TestUnsqueeze2Op1_AxesTensor(TestUnsqueeze2Op_AxesTensor):

        def init_test_case(self):
            if False:
                return 10
            self.ori_shape = (20, 5)
            self.axes = (-1,)
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2_AxesTensor(TestUnsqueeze2Op_AxesTensor):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    class TestUnsqueeze2Op3_AxesTensor(TestUnsqueeze2Op_AxesTensor):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    class TestUnsqueeze2Op4_AxesTensor(TestUnsqueeze2Op_AxesTensor):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)
support_types = get_xpu_op_support_types('unsqueeze2')
for stype in support_types:
    create_test_class(globals(), XPUTestUnsqueeze2Op, stype)
if __name__ == '__main__':
    unittest.main()