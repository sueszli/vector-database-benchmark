import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op import Operator
from op_test_xpu import XPUOpTest
import paddle
from paddle.base import core
paddle.enable_static()

class XPUTestShapeOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'shape'
        self.use_dynamic_create_class = False

    class TestShapeOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type
            self.op_type = 'shape'
            self.config()
            input = np.zeros(self.shape)
            self.inputs = {'Input': input.astype(self.dtype)}
            self.outputs = {'Out': np.array(self.shape)}

        def config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = [2, 3]

        def test_check_output(self):
            if False:
                return 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestShapeOp1(TestShapeOp):

        def config(self):
            if False:
                i = 10
                return i + 15
            self.shape = [2]

    class TestShapeOp2(TestShapeOp):

        def config(self):
            if False:
                while True:
                    i = 10
            self.shape = [1, 2, 3]

    class TestShapeOp3(TestShapeOp):

        def config(self):
            if False:
                while True:
                    i = 10
            self.shape = [1, 2, 3, 4]

    class TestShapeOp4(TestShapeOp):

        def config(self):
            if False:
                while True:
                    i = 10
            self.shape = [1, 2, 3, 4, 1024]

    class TestShapeOp5(TestShapeOp):

        def config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = [1, 2, 3, 4, 1, 201]

    class TestShapeWithSelectedRows(unittest.TestCase):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type

        def get_places(self):
            if False:
                for i in range(10):
                    print('nop')
            return [core.CPUPlace(), core.XPUPlace(0)]

        def check_with_place(self, place):
            if False:
                return 10
            scope = core.Scope()
            x_rows = [0, 1, 5, 4, 19]
            height = 20
            row_numel = 2
            np_array = np.ones((len(x_rows), row_numel)).astype(self.dtype)
            x = scope.var('X').get_selected_rows()
            x.set_rows(x_rows)
            x.set_height(height)
            x_tensor = x.get_tensor()
            x_tensor.set(np_array, place)
            out_shape = scope.var('Out').get_tensor()
            op = Operator('shape', Input='X', Out='Out')
            op.run(scope, place)
            out_shape = np.array(out_shape).tolist()
            self.assertListEqual([5, 2], out_shape)

        def test_check_output(self):
            if False:
                return 10
            for place in self.get_places():
                if type(place) is paddle.base.libpaddle.CPUPlace and self.dtype == np.float16:
                    pass
                else:
                    self.check_with_place(place)
support_types = get_xpu_op_support_types('shape')
for stype in support_types:
    create_test_class(globals(), XPUTestShapeOp, stype)
if __name__ == '__main__':
    unittest.main()