import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def random_unique_float(shape, dtype):
    if False:
        return 10
    numel = np.prod(shape)
    arr = np.random.uniform(-10.0, 10.0, numel * 10).astype(dtype)
    arr = np.unique(arr)
    assert arr.shape[0] >= numel, 'failed to create enough unique values: %d vs %d' % (arr.shape[0], numel)
    arr = arr[:numel]
    np.random.shuffle(arr)
    arr = arr.reshape(shape)
    return arr

def numpy_topk(x, k=1, axis=-1, largest=True):
    if False:
        print('Hello World!')
    if axis < 0:
        axis = len(x.shape) + axis
    if largest:
        indices = np.argsort(-x, axis=axis)
    else:
        indices = np.argsort(x, axis=axis)
    if largest:
        value = -np.sort(-x, axis=axis)
    else:
        value = np.sort(x, axis=axis)
    indices = indices.take(indices=range(0, k), axis=axis)
    value = value.take(indices=range(0, k), axis=axis)
    return (value, indices)

class XPUTestTopKV2Op(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'top_k_v2'
        self.use_dynamic_create_class = False

    class TestTopkOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.op_type = 'top_k_v2'
            self.dtype = self.in_type
            self.init_args()
            self.input_data = random_unique_float(self.input_data_shape, self.dtype)
            self.inputs = {'X': self.input_data}
            self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
            (output, indices) = numpy_topk(self.input_data, axis=self.axis, k=self.k, largest=self.largest)
            self.outputs = {'Out': output, 'Indices': indices}

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                return 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_args(self):
            if False:
                print('Hello World!')
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 20)

    class TestTopkOp1(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 3
            self.axis = 1
            self.largest = True
            if self.dtype == np.float16:
                self.input_data_shape = (100, 55)
            else:
                self.input_data_shape = (100, 155)

    class TestTopkOp2(TestTopkOp):

        def init_args(self):
            if False:
                i = 10
                return i + 15
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp3(TestTopkOp):

        def init_args(self):
            if False:
                print('Hello World!')
            self.k = 5
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp4(TestTopkOp):

        def init_args(self):
            if False:
                for i in range(10):
                    print('nop')
            self.k = 1
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp5(TestTopkOp):

        def init_args(self):
            if False:
                print('Hello World!')
            self.k = 3
            self.axis = 2
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp6(TestTopkOp):

        def init_args(self):
            if False:
                for i in range(10):
                    print('nop')
            self.k = 5
            self.axis = 1
            self.largest = True
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkOp7(TestTopkOp):

        def init_args(self):
            if False:
                print('Hello World!')
            self.k = 10
            self.axis = 2
            self.largest = True
            self.input_data_shape = (8, 5, 10, 16)

    class TestTopkOp8(TestTopkOp):

        def init_args(self):
            if False:
                for i in range(10):
                    print('nop')
            self.k = 1
            self.axis = 1
            self.largest = True
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkOp9(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp10(TestTopkOp):

        def init_args(self):
            if False:
                i = 10
                return i + 15
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp11(TestTopkOp):

        def init_args(self):
            if False:
                for i in range(10):
                    print('nop')
            self.k = 5
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp12(TestTopkOp):

        def init_args(self):
            if False:
                while True:
                    i = 10
            self.k = 1
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp1(TestTopkOp):

        def init_args(self):
            if False:
                while True:
                    i = 10
            self.k = 3
            self.axis = 1
            self.largest = False
            if self.dtype == np.float16:
                self.input_data_shape = (100, 55)
            else:
                self.input_data_shape = (100, 155)

    class TestTopkSmallestOp2(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 3
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp3(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 5
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp4(TestTopkOp):

        def init_args(self):
            if False:
                i = 10
                return i + 15
            self.k = 1
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp5(TestTopkOp):

        def init_args(self):
            if False:
                print('Hello World!')
            self.k = 3
            self.axis = 2
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp6(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 5
            self.axis = 1
            self.largest = False
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkSmallestOp7(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 10
            self.axis = 2
            self.largest = False
            self.input_data_shape = (8, 5, 10, 16)

    class TestTopkSmallestOp8(TestTopkOp):

        def init_args(self):
            if False:
                i = 10
                return i + 15
            self.k = 1
            self.axis = 1
            self.largest = False
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkSmallestOp9(TestTopkOp):

        def init_args(self):
            if False:
                for i in range(10):
                    print('nop')
            self.k = 3
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp10(TestTopkOp):

        def init_args(self):
            if False:
                while True:
                    i = 10
            self.k = 3
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp11(TestTopkOp):

        def init_args(self):
            if False:
                return 10
            self.k = 5
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp12(TestTopkOp):

        def init_args(self):
            if False:
                i = 10
                return i + 15
            self.k = 1
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)
support_types = get_xpu_op_support_types('top_k_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestTopKV2Op, stype)
if __name__ == '__main__':
    unittest.main()