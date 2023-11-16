import os
import sys
import unittest
import numpy as np
import paddle
import paddle.static
from paddle.utils.cpp_extension import load
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from op_test_ipu import IPUOpTest

def load_custom_ops():
    if False:
        print('Hello World!')
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(name='custom_nll_loss', sources=[f'{cur_dir}/custom_nllloss.cc'], extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'], extra_ldflags=['-lpopfloat'])
    return custom_ops

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.load_custom_ops()
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    @property
    def fp16_enabled(self):
        if False:
            return 10
        return False

    def load_custom_ops(self):
        if False:
            i = 10
            return i + 15
        self.custom_ops = load_custom_ops()

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        x = np.random.rand(16, 20, 256).astype('float32')
        label = np.random.uniform(0, 256, size=[16, 20]).astype('int32')
        self.feed_fp32 = {'x': x, 'label': label}

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = self.custom_ops.custom_nll_loss
        self.op_attrs = {'reduction': 0, 'ignoreindex': '0', 'inputislogprobability': False}

    def set_feed_attr(self):
        if False:
            while True:
                i = 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        label = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32')
        out = self.op(x, label, **self.op_attrs)
        out = paddle.mean(out)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            print('Hello World!')
        self.build_model()
        self.run_model(IPUOpTest.ExecutionMode.IPU_FP32)
        print(self.output_dict)

class TestCase1(TestBase):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = self.custom_ops.custom_nll_loss
        self.op_attrs = {'reduction': 0, 'ignoreindex': 'None', 'inputislogprobability': False}
if __name__ == '__main__':
    unittest.main()