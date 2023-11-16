import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.nn.functional as F
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            return 10
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_test_op(self):
        if False:
            return 10
        self.op = F.elu
        self.op_attrs = {}

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        self.feed_list = list(self.feed_fp32.keys())

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = self.op(x, **self.op_attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestHardTanhCase0(TestBase):

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[1, 3, 10, 10]) * 30
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        self.feed_list = list(self.feed_fp32.keys())

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = paddle.nn.functional.hardtanh
        self.op_attrs = {}

class TestHardTanhCase1(TestHardTanhCase0):

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.nn.functional.hardtanh
        self.op_attrs = {'min': 0.1, 'max': 10.0}

class TestEluCase1(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.elu
        self.op_attrs = {'alpha': 0.3}

class TestHardShrinkCase0(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.hardshrink
        self.op_attrs = {}

class TestHardSigmoidCase0(TestBase):

    def set_test_op(self):
        if False:
            print('Hello World!')
        self.op = F.hardsigmoid
        self.op_attrs = {}

class TestHardSigmoidCase1(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.hardsigmoid
        self.op_attrs = {'slope': 0.2, 'offset': 0.33}

class TestHardSwishCase0(TestBase):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = F.hardswish
        self.op_attrs = {}

class TestLeakyReluCase0(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.leaky_relu
        self.op_attrs = {}

class TestLeakyReluCase1(TestBase):

    def set_test_op(self):
        if False:
            return 10
        self.op = F.leaky_relu
        self.op_attrs = {'negative_slope': 0.2333}

class TestLog10Case0(TestBase):

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.log10
        self.op_attrs = {}

class TestLog1pCase0(TestBase):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.log1p
        self.op_attrs = {}

class TestLog2Case0(TestBase):

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.log2
        self.op_attrs = {}

class TestLogSigmoidCase0(TestBase):

    def set_test_op(self):
        if False:
            return 10
        self.op = F.log_sigmoid
        self.op_attrs = {}

class TestLogSoftmaxCase0(TestBase):

    def set_test_op(self):
        if False:
            return 10
        self.op = F.log_softmax
        self.op_attrs = {}

class TestMishCase0(TestBase):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = F.mish
        self.op_attrs = {}

class TestRelu6Case0(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.relu6
        self.op_attrs = {}

class TestRsqrtCase0(TestBase):

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.rsqrt
        self.op_attrs = {}

class TestSeluCase0(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.selu
        self.op_attrs = {}

class TestSiluCase0(TestBase):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = F.silu
        self.op_attrs = {}

class TestSoftShrinkCase0(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.softshrink
        self.op_attrs = {}

class TestSoftShrinkCase1(TestBase):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = F.softshrink
        self.op_attrs = {'threshold': 0.2333}

class TestSquareCase0(TestBase):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.square
        self.op_attrs = {}

class TestSwishCase0(TestBase):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = F.swish
        self.op_attrs = {}

class TestTanhShrinkCase0(TestBase):

    def set_atol(self):
        if False:
            i = 10
            return i + 15
        super().set_atol()
        self.atol = 1e-07

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = F.tanhshrink
        self.op_attrs = {}

class TestThresholdedReluCase0(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.thresholded_relu
        self.op_attrs = {}

class TestThresholdedReluCase1(TestBase):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = F.thresholded_relu
        self.op_attrs = {'threshold': 0.2333}
if __name__ == '__main__':
    unittest.main()