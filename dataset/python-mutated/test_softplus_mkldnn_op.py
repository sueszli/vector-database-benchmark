import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle

def ref_softplus(x, beta, threshold):
    if False:
        print('Hello World!')
    x_beta = beta * x
    out = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    return out

@OpTestTool.skip_if_not_cpu_bf16()
class TestSoftplusOneDNNOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'softplus'
        self.beta = 1
        self.threshold = 20
        self.config()
        self.set_dtype()
        self.attrs = {'use_mkldnn': True, 'beta': self.beta}
        self.x = np.random.random(self.x_shape)
        self.out = ref_softplus(self.x, self.beta, self.threshold)
        if self.dtype != np.float32:
            self.x = convert_float_to_uint16(self.x)
        self.inputs = {'X': self.out}
        self.outputs = {'Out': ref_softplus(self.out, self.beta, self.threshold)}

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = (10, 10)

    def set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

class TestSoftplus4DOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.x_shape = (10, 5, 4, 2)

class TestSoftplus6DOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = (3, 2, 2, 5, 4, 2)

class TestSoftplus6DExtendedFunctorOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.x_shape = (3, 5, 2, 5, 4, 2)
        self.beta = 2.5

class TestSoftplus3DExtendedFunctorOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        if False:
            return 10
        self.x_shape = (20, 4, 2)
        self.beta = 0.4

class TestSoftplusBF16OneDNNOp(TestSoftplusOneDNNOp):

    def set_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.uint16

class TestSoftplus4DBF16OneDNNOp(TestSoftplus4DOneDNNOp):

    def set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

class TestSoftplus6DBF16OneDNNOp(TestSoftplus6DOneDNNOp):

    def set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

class TestSoftplus3DExtendedFunctorBF16OneDNNOp(TestSoftplus3DExtendedFunctorOneDNNOp):

    def set_dtype(self):
        if False:
            return 10
        self.dtype = np.uint16
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()