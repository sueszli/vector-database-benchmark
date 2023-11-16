import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def output_hist(out):
    if False:
        for i in range(10):
            print('nop')
    (hist, _) = np.histogram(out, range=(-10, 10))
    hist = hist.astype('float32')
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return (hist, prob)

class XPUTestRandIntOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'randint'
        self.use_dynamic_create_class = False

    class TestXPURandIntOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'randint'
            self.dtype = self.in_type
            self.set_attrs()
            self.atol = 0.0001
            np.random.seed(10)
            self.inputs = {}
            self.outputs = {'Out': np.zeros((10000, 784)).astype('float32')}
            self.attrs = {'shape': [10000, 784], 'low': -10, 'high': 10, 'seed': 10}
            self.output_hist = output_hist

        def set_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_customized(self.verify_output)

        def verify_output(self, outs):
            if False:
                print('Hello World!')
            (hist, prob) = self.output_hist(np.array(outs[0]))
            np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)
support_types = get_xpu_op_support_types('randint')
for stype in support_types:
    create_test_class(globals(), XPUTestRandIntOp, stype)
if __name__ == '__main__':
    unittest.main()