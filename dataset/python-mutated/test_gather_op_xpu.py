import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def gather_numpy(x, index, axis):
    if False:
        i = 10
        return i + 15
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather

class XPUTestGather(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'gather'

    class TestXPUGatherOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'gather'
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_config()
            xnp = np.random.random(self.x_shape).astype(self.dtype)
            self.inputs = {'X': xnp, 'Index': np.array(self.index).astype(self.index_type)}
            self.outputs = {'Out': self.inputs['X'][self.inputs['Index']]}

        def init_config(self):
            if False:
                return 10
            self.x_shape = (10, 20)
            self.index = [1, 3, 5]
            self.index_type = np.int32

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            if paddle.is_compiled_with_xpu():
                self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            if paddle.is_compiled_with_xpu():
                self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestCase1(TestXPUGatherOp):

        def init_config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = 100
            self.index = [1, 3, 5]
            self.index_type = np.int32

    class TestCase2(TestXPUGatherOp):

        def init_config(self):
            if False:
                return 10
            self.x_shape = 100
            self.index = [1, 3, 5]
            self.index_type = np.int64

    class TestCase3(TestXPUGatherOp):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = (10, 20)
            self.index = [1, 3, 5]
            self.index_type = np.int32

    class TestCase4(TestXPUGatherOp):

        def init_config(self):
            if False:
                return 10
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': False}
            self.index = [1, 1]
            self.index_type = np.int32

    class TestCase5(TestXPUGatherOp):

        def init_config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': False}
            self.index = [1, 1, 3]
            self.index_type = np.int32

    class TestCase6(TestXPUGatherOp):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': True}
            self.index = [1, 3]
            self.index_type = np.int32

    class TestCase7(TestXPUGatherOp):

        def init_config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': True}
            self.index = [1, 3]
            self.index_type = np.int64

class TestGatherOpEmpty(unittest.TestCase):

    def test_gather_empty_index(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_xpu():
            paddle.set_device('xpu')
            paddle.disable_static()
            data = paddle.ones([10], dtype='int32')
            index = paddle.ones([], dtype='int32')
            out = paddle.gather(data, index)
            self.assertEqual(out.shape, index.shape)
            paddle.enable_static()
support_types = get_xpu_op_support_types('gather')
for stype in support_types:
    create_test_class(globals(), XPUTestGather, stype)
if __name__ == '__main__':
    unittest.main()