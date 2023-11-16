import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestAccuracyOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'accuracy'
        self.use_dynamic_create_class = False

    class TestXPUAccuracyOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'accuracy'
            self.init_dtype()
            n = 8192
            infer = np.random.random((n, 1)).astype(self.dtype)
            indices = np.random.randint(0, 2, (n, 1)).astype('int64')
            label = np.random.randint(0, 2, (n, 1)).astype('int64')
            self.inputs = {'Out': infer, 'Indices': indices, 'Label': label}
            num_correct = 0
            for rowid in range(n):
                for ele in indices[rowid]:
                    if ele == label[rowid]:
                        num_correct += 1
                        break
            self.outputs = {'Accuracy': np.array(num_correct / float(n)).astype(self.dtype), 'Correct': np.array(num_correct).astype('int32'), 'Total': np.array(n).astype('int32')}
            self.attrs = {'use_xpu': True}

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                print('Hello World!')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)
support_types = get_xpu_op_support_types('accuracy')
for stype in support_types:
    create_test_class(globals(), XPUTestAccuracyOp, stype)
if __name__ == '__main__':
    unittest.main()