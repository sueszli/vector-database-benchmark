import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestSequenceUnpadOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'sequence_unpad'
        self.use_dynamic_create_class = False

    class TestSequenceUnpadOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.init_dtype()
            self.initTestCase()
            self.set_xpu()
            self.op_type = 'sequence_unpad'
            self.place = paddle.XPUPlace(0)
            self.compute()

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def set_xpu(self):
            if False:
                i = 10
                return i + 15
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(self.place)

        def initTestCase(self):
            if False:
                return 10
            self.length = [2, 3, 4]
            self.x_shape = (3, 40)

        def compute(self):
            if False:
                for i in range(10):
                    print('nop')
            assert len(self.length) == self.x_shape[0]
            x = np.random.random(self.x_shape).astype(self.dtype)
            out_lod = [self.length]
            out = x[0, 0:self.length[0]]
            for i in range(1, x.shape[0]):
                out = np.append(out, x[i, 0:self.length[i]], axis=0)
            out_shape = (sum(self.length),)
            if len(self.x_shape) == 2:
                out_shape = out_shape + (1,)
            else:
                out_shape = out_shape + self.x_shape[2:]
            self.inputs = {'X': x, 'Length': np.array(self.length).astype('int64')}
            self.outputs = {'Out': (out.reshape(out_shape), out_lod)}

    class TestSequenceUnpadOp2(TestSequenceUnpadOp):

        def initTestCase(self):
            if False:
                return 10
            self.length = [2, 3, 4]
            self.x_shape = (3, 5, 4, 3)

    class TestSequenceUnpadOp3(TestSequenceUnpadOp):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.length = [5, 2, 3, 4]
            self.x_shape = (4, 5, 3, 3, 6)

    class TestSequenceUnpadOp4(TestSequenceUnpadOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.length = [5, 5, 5, 5]
            self.x_shape = (4, 5, 3, 3, 6)

    class TestSequenceUnpadOp5(TestSequenceUnpadOp):

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.length = [1, 4, 3, 1]
            self.x_shape = (4, 5, 3, 3, 6)

class TestSequenceUnpadOpError(unittest.TestCase):

    def test_error(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The type of 'x' in paddle.static.nn.sequence_unpad must be <class 'paddle.base.framework.Variable'>, but received <class 'numpy.ndarray'>.\n        "

        def test_x_variable():
            if False:
                for i in range(10):
                    print('nop')
            x = np.random.random((10, 5)).astype('float64')
            len = paddle.static.data(name='length2', shape=[10], dtype='int64')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x, length=len)
        self.assertRaises(TypeError, test_x_variable)
        "\n        The type of 'length' in base.layers.sequence_unpad must be <class 'paddle.base.framework.Variable'>, but received <class 'numpy.ndarray'>.\n        "

        def test_length_variable():
            if False:
                return 10
            x1 = paddle.static.data(name='x1', shape=[10, 5], dtype='float32')
            len1 = np.random.random(10).astype('int64')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x1, length=len1)
        self.assertRaises(TypeError, test_length_variable)
        "\n        The data type of 'x' in base.layers.sequence_unpad must be ['float32', 'float64', 'int32', 'int64'], but received float16\n        "

        def test_x_dtype():
            if False:
                return 10
            x2 = paddle.static.data(name='x2', shape=[10, 5], dtype='float16')
            len2 = paddle.static.data(name='length2', shape=[10], dtype='int64')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x2, length=len2)
        self.assertRaises(TypeError, test_x_dtype)
        "\n        The data type of 'length' in base.layers.sequence_unpad must be ['int64'], but received int32\n        "

        def test_length_dtype():
            if False:
                i = 10
                return i + 15
            x3 = paddle.static.data(name='x3', shape=[10, 5], dtype='float64')
            len3 = paddle.static.data(name='length3', shape=[10], dtype='int32')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x3, length=len3)
        self.assertRaises(TypeError, test_length_dtype)
support_types = get_xpu_op_support_types('sequence_unpad')
for stype in support_types:
    create_test_class(globals(), XPUTestSequenceUnpadOp, stype)
if __name__ == '__main__':
    unittest.main()