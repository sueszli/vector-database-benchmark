import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def strided_slice_native_forward(input, axes, starts, ends, strides):
    if False:
        i = 10
        return i + 15
    dim = input.ndim
    start = []
    end = []
    stride = []
    for i in range(dim):
        start.append(0)
        end.append(input.shape[i])
        stride.append(1)
    for i in range(len(axes)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        stride[axes[i]] = strides[i]
    result = {1: lambda input, start, end, stride: input[start[0]:end[0]:stride[0]], 2: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], start[1]:end[1]:stride[1]], 3: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], start[1]:end[1]:stride[1], start[2]:end[2]:stride[2]], 4: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3]], 5: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3], start[4]:end[4]:stride[4]], 6: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3], start[4]:end[4]:stride[4], start[5]:end[5]:stride[5]]}[dim](input, start, end, stride)
    return result

class XPUTestStrideSliceOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'strided_slice'
        self.use_dynamic_create_class = False

    class XPUTestStrideSliceOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'strided_slice'
            self.dtype = self.in_type
            self.initTestCase()
            self.input = np.random.random(self.inshape).astype(self.dtype)
            self.python_api = paddle.strided_slice
            self.output = strided_slice_native_forward(self.input, self.axes, self.starts, self.ends, self.strides)
            self.inputs = {'Input': self.input}
            self.outputs = {'Out': self.output}
            self.attrs = {'axes': self.axes, 'starts': self.starts, 'ends': self.ends, 'strides': self.strides, 'infer_flags': self.infer_flags}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(paddle.XPUPlace(0), ['Input'], 'Out')

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inshape = 100
            self.axes = [0]
            self.starts = [-4]
            self.ends = [-1]
            self.strides = [1]
            self.infer_flags = [1]

    class XPUTestStrideSliceOp1(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.inshape = 100
            self.axes = [0]
            self.starts = [3]
            self.ends = [8]
            self.strides = [1]
            self.infer_flags = [1]

    class XPUTestStrideSliceOp2(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                return 10
            self.inshape = (4, 8, 12)
            self.axes = [0, 1, 2]
            self.starts = [3, 4, 5]
            self.ends = [4, 5, 6]
            self.strides = [1, 1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp3(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                return 10
            self.inshape = (4, 8, 12, 4, 40)
            self.axes = [0, 1, 2, 3, 4]
            self.starts = [3, 4, 5, 1, 10]
            self.ends = [4, 5, 6, 2, 30]
            self.strides = [1, 1, 1, 2, 2]
            self.infer_flags = [1, 1, 1, 1, 1]

    class XPUTestStrideSliceOp5(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inshape = (5, 5, 5)
            self.axes = [0, 1, 2]
            self.starts = [1, 0, 0]
            self.ends = [2, 1, 3]
            self.strides = [1, 1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp7(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.inshape = (5, 5, 5)
            self.axes = [0, 1, 2]
            self.starts = [1, 0, 0]
            self.ends = [2, 2, 3]
            self.strides = [1, 1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp8(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                return 10
            self.inshape = (3, 3, 3, 6, 7, 8)
            self.axes = [0, 1, 2, 3, 4, 5]
            self.starts = [1, 0, 0, 0, 1, 2]
            self.ends = [2, 2, 3, 1, 2, 8]
            self.strides = [1, 1, 1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1, 1]

    class XPUTestStrideSliceOp_eb_1(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.inshape = (1, 4, 4096, 128)
            self.axes = [0, 1, 2, 3]
            self.starts = [0, 0, 0, 0]
            self.ends = [1, 4, 4096, 128]
            self.strides = [1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1]

    class XPUTestStrideSliceOp_eb_2(XPUTestStrideSliceOp):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.inshape = (1, 4, 4096, 128)
            self.axes = [0, 1, 2, 3]
            self.starts = [0, 0, 0, 1]
            self.ends = [1, 4, 4096, 128]
            self.strides = [1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1]
support_types = get_xpu_op_support_types('strided_slice')
for stype in support_types:
    create_test_class(globals(), XPUTestStrideSliceOp, stype)
if __name__ == '__main__':
    unittest.main()