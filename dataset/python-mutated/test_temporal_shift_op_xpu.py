import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
import paddle.nn.functional as F
paddle.enable_static()
np.random.seed(10)

def temporal_shift(x, seg_num, shift_ratio, data_format):
    if False:
        return 10
    if data_format == 'NHWC':
        x = np.transpose(x, (0, 3, 1, 2))
    shape = x.shape
    reshape_x = x.reshape((-1, seg_num, shape[1], shape[2], shape[3]))
    pad_x = np.pad(reshape_x, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)), 'constant')
    c1 = int(shape[1] * shift_ratio)
    c2 = int(shape[1] * 2 * shift_ratio)
    slice1 = pad_x[:, :seg_num, :c1, :, :]
    slice2 = pad_x[:, 2:seg_num + 2, c1:c2, :, :]
    slice3 = pad_x[:, 1:seg_num + 1, c2:, :, :]
    concat_x = np.concatenate([slice1, slice2, slice3], axis=2)
    out = concat_x.reshape(shape)
    if data_format == 'NHWC':
        out = np.transpose(out, (0, 2, 3, 1))
    return out

class XPUTestTemporalShiftOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'temporal_shift'
        self.use_dynamic_create_class = False

    class TestXPUTemporalShift(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.initTestCase()
            self.op_type = 'temporal_shift'
            self.python_api = F.temporal_shift
            self.use_xpu = True
            x = np.random.random(self.x_shape).astype(self.dtype)
            self.attrs = {'seg_num': self.seg_num, 'shift_ratio': self.shift_ratio, 'data_format': self.data_format}
            self.inputs = {'X': x}
            output = temporal_shift(x, self.seg_num, self.shift_ratio, self.data_format)
            self.outputs = {'Out': output}
            self.python_out_sig = ['Out']

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output(check_dygraph=False)

        def test_check_grad(self):
            if False:
                return 10
            self.check_grad(['X'], 'Out', check_dygraph=False)

        def initTestCase(self):
            if False:
                return 10
            self.x_shape = (6, 4, 4, 4)
            self.seg_num = 3
            self.shift_ratio = 0.25
            self.dtype = 'float32'
            self.data_format = 'NCHW'

    class TestXPUTemporalShift2(TestXPUTemporalShift):

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = (1, 1, 1, 1)
            self.seg_num = 1
            self.shift_ratio = 0.1
            self.dtype = 'float32'
            self.data_format = 'NCHW'

    class TestXPUTemporalShift3(TestXPUTemporalShift):

        def initTestCase(self):
            if False:
                return 10
            self.x_shape = (4, 9, 1, 1)
            self.seg_num = 2
            self.shift_ratio = 0.2
            self.dtype = 'float32'
            self.data_format = 'NCHW'

    class TestXPUTemporalShift4(TestXPUTemporalShift):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.x_shape = (4, 1, 10, 10)
            self.seg_num = 2
            self.shift_ratio = 0.3
            self.dtype = 'float32'
            self.data_format = 'NCHW'

    class TestXPUTemporalShift5(TestXPUTemporalShift):

        def initTestCase(self):
            if False:
                while True:
                    i = 10
            self.x_shape = (1, 1, 1, 1)
            self.seg_num = 1
            self.shift_ratio = 0.3
            self.dtype = 'float32'
            self.data_format = 'NHWC'

    class TestXPUTemporalShift6(TestXPUTemporalShift):

        def initTestCase(self):
            if False:
                while True:
                    i = 10
            self.x_shape = (6, 5, 5, 1)
            self.seg_num = 3
            self.shift_ratio = 0.25
            self.dtype = 'float32'
            self.data_format = 'NHWC'

    class TestXPUTemporalShift7(TestXPUTemporalShift):

        def initTestCase(self):
            if False:
                while True:
                    i = 10
            self.x_shape = (9, 1, 1, 4)
            self.seg_num = 3
            self.shift_ratio = 0.45
            self.dtype = 'float32'
            self.data_format = 'NHWC'
support_types = get_xpu_op_support_types('temporal_shift')
for stype in support_types:
    create_test_class(globals(), XPUTestTemporalShiftOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()